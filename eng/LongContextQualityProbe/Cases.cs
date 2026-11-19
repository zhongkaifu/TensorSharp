// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The case file eng/validation/long-context-quality.py writes (cases.jsonl: a header line naming the pinned corpora, then
// one case a line), the corpora it names, and the token inputs of each case: corpus windows, prompt windows ending at a
// chosen character, and chat prompts whose document is fitted to an exact token count with the model's own tokenizer and
// chat template.
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using TensorSharp.Runtime;

namespace LongContextQualityProbe;

internal sealed class Insert
{
    public double Depth;
    public string Text;
}

internal sealed class Case
{
    public string Id, Kind, Group, Corpus;
    public int Tokens, ScoreTokens, NewTokens, StartToken, EndChar, FitTokens, MaxNewTokens, StartParagraph;
    public bool Think;
    public List<(string Role, string Content)> Messages = new();
    public List<Insert> Inserts = new();
    public bool HasDocument;

    /// <summary>The context a case occupies, for --max-tokens and the out-of-memory skip: its window or fitted prompt plus what it
    /// generates.</summary>
    public int Footprint => Kind == "chat" ? FitTokens + MaxNewTokens : Tokens + NewTokens;

    public static Case Parse(JsonNode node)
    {
        var c = new Case
        {
            Id = Str(node, "id") ?? throw new FormatException("a case has no id"),
            Kind = Str(node, "kind") ?? throw new FormatException("a case has no kind"),
            Group = Str(node, "group"),
            Corpus = Str(node, "corpus"),
            Tokens = Int(node, "tokens"),
            ScoreTokens = Int(node, "score_tokens"),
            NewTokens = Int(node, "new_tokens"),
            StartToken = Int(node, "start_token"),
            EndChar = Int(node, "end_char"),
            FitTokens = Int(node, "fit_tokens"),
            MaxNewTokens = Int(node, "max_new_tokens"),
            Think = node["think"]?.GetValue<bool>() ?? false,
        };
        if (node["messages"] is JsonArray messages)
            foreach (JsonNode m in messages)
                c.Messages.Add((Str(m, "role"), Str(m, "content")));
        if (node["document"] is JsonObject doc)
        {
            c.HasDocument = true;
            c.Corpus = Str(doc, "corpus") ?? throw new FormatException($"{c.Id}: the document names no corpus");
            c.StartParagraph = Int(doc, "start_paragraph");
            if (doc["inserts"] is JsonArray inserts)
                foreach (JsonNode i in inserts)
                    c.Inserts.Add(new Insert { Depth = i["depth"].GetValue<double>(), Text = Str(i, "text") });
            c.Inserts.Sort((a, b) => a.Depth.CompareTo(b.Depth));
        }
        switch (c.Kind)
        {
            case "ppl" or "continue":
                if (c.Corpus == null || c.Tokens <= 0 || c.ScoreTokens <= 0 || c.ScoreTokens >= c.Tokens)
                    throw new FormatException($"{c.Id}: a {c.Kind} case needs a corpus and 0 < score_tokens < tokens");
                break;
            case "layers":
                if (c.Corpus == null || c.Tokens <= 0) throw new FormatException($"{c.Id}: a layers case needs a corpus and tokens");
                break;
            case "greedy":
                if (c.Corpus == null || c.Tokens <= 0 || c.NewTokens <= 0 || c.EndChar <= 0)
                    throw new FormatException($"{c.Id}: a greedy case needs a corpus, tokens, end_char and new_tokens");
                break;
            case "chat":
                if (c.Messages.Count == 0 || c.MaxNewTokens <= 0) throw new FormatException($"{c.Id}: a chat case needs messages and max_new_tokens");
                if (c.HasDocument != (c.FitTokens > 0)) throw new FormatException($"{c.Id}: a document needs fit_tokens and fit_tokens a document");
                if (c.HasDocument && !c.Messages.Any(m => m.Content.Contains(DocumentSlot, StringComparison.Ordinal)))
                    throw new FormatException($"{c.Id}: no message holds {DocumentSlot}");
                break;
            default:
                throw new FormatException($"{c.Id}: unknown kind '{c.Kind}' (ppl, continue, layers, greedy, chat)");
        }
        return c;
    }

    public const string DocumentSlot = "{{document}}";

    private static string Str(JsonNode n, string key) => n[key]?.GetValue<string>();
    private static int Int(JsonNode n, string key) => n[key]?.GetValue<int>() ?? 0;
}

/// <summary>The case file: its header (the corpora with their sha256), its cases, and the corpora's text and tokens.</summary>
internal sealed class CaseFile
{
    public string Path, Sha256;
    public JsonObject Header;
    public List<Case> Cases = new();
    private readonly Dictionary<string, string> _corpusText = new();
    private readonly Dictionary<string, int[]> _corpusTokens = new();
    private readonly Dictionary<string, string[]> _paragraphs = new();

    public static CaseFile Load(string path)
    {
        var file = new CaseFile { Path = System.IO.Path.GetFullPath(path) };
        byte[] bytes = File.ReadAllBytes(file.Path);
        file.Sha256 = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
        var ids = new HashSet<string>(StringComparer.Ordinal);
        foreach (string line in Encoding.UTF8.GetString(bytes).Split('\n'))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            JsonNode node = JsonNode.Parse(line);
            string type = node["type"]?.GetValue<string>();
            if (type == "header") file.Header = node.AsObject();
            else if (type == "case")
            {
                Case c = Case.Parse(node);
                if (!ids.Add(c.Id)) throw new FormatException($"case id '{c.Id}' appears twice");
                file.Cases.Add(c);
            }
        }
        if (file.Header == null) throw new FormatException($"{path} has no header line (written by eng/validation/long-context-quality.py cases)");
        foreach (Case c in file.Cases)
            if (c.Corpus != null && file.Header["corpora"]?[c.Corpus] == null)
                throw new FormatException($"{c.Id}: corpus '{c.Corpus}' is not in the header");
        return file;
    }

    /// <summary>A corpus's text, read once and checked against the sha256 the header pins.</summary>
    public string CorpusText(string name)
    {
        if (_corpusText.TryGetValue(name, out string text)) return text;
        JsonNode entry = Header["corpora"][name];
        string file = System.IO.Path.Combine(System.IO.Path.GetDirectoryName(Path), entry["file"].GetValue<string>());
        byte[] bytes = File.ReadAllBytes(file);
        string sha = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
        string pinned = entry["sha256"].GetValue<string>();
        if (sha != pinned) throw new InvalidDataException($"corpus '{name}' ({file}) has sha256 {sha}, the case file pins {pinned}");
        text = Encoding.UTF8.GetString(bytes);
        _corpusText[name] = text;
        return text;
    }

    /// <summary>A whole corpus as tokens (no special tokens), encoded once.</summary>
    public int[] CorpusTokens(string name, ITokenizer tokenizer)
    {
        if (_corpusTokens.TryGetValue(name, out int[] tokens)) return tokens;
        tokens = tokenizer.Encode(CorpusText(name), addSpecial: false).ToArray();
        _corpusTokens[name] = tokens;
        return tokens;
    }

    public string[] Paragraphs(string name)
    {
        if (_paragraphs.TryGetValue(name, out string[] p)) return p;
        p = CorpusText(name).Split("\n\n", StringSplitOptions.RemoveEmptyEntries);
        _paragraphs[name] = p;
        return p;
    }

    /// <summary>The <see cref="Case.Tokens"/> corpus tokens of a window case from <see cref="Case.StartToken"/>.</summary>
    public int[] Window(Case c, ITokenizer tokenizer)
    {
        int[] all = CorpusTokens(c.Corpus, tokenizer);
        if ((long)c.StartToken + c.Tokens > all.Length)
            throw new InvalidDataException($"{c.Id}: corpus '{c.Corpus}' has {all.Length} tokens, the window needs [{c.StartToken}, {c.StartToken + c.Tokens})");
        return all.AsSpan(c.StartToken, c.Tokens).ToArray();
    }

    /// <summary>The last <see cref="Case.Tokens"/> tokens of the corpus text ending at <see cref="Case.EndChar"/> (a greedy
    /// prompt ends where the case builder cut it, at a function boundary; its first token may start mid-word, which is dropped).</summary>
    public int[] PromptEndingAt(Case c, ITokenizer tokenizer)
    {
        string text = CorpusText(c.Corpus);
        if (c.EndChar > text.Length) throw new InvalidDataException($"{c.Id}: end_char {c.EndChar} is past the corpus ({text.Length} chars)");
        for (int chars = c.Tokens * 6; ; chars *= 2)
        {
            int start = Math.Max(0, c.EndChar - chars);
            List<int> tokens = tokenizer.Encode(text[start..c.EndChar], addSpecial: false);
            if (tokens.Count > c.Tokens) return tokens.GetRange(tokens.Count - c.Tokens, c.Tokens).ToArray();
            if (start == 0) throw new InvalidDataException($"{c.Id}: only {tokens.Count} tokens before end_char {c.EndChar}, the case needs {c.Tokens}");
        }
    }

    /// <summary>
    /// A chat case's prompt tokens. With a document, the {{document}} slot holds the longest prefix of the corpus's paragraphs
    /// from <see cref="Case.StartParagraph"/> (cut at a character, the inserts placed at their depth fractions of the cut text)
    /// for which the whole rendered prompt (the model's chat template, generation prompt included) has at most
    /// <see cref="Case.FitTokens"/> tokens; it must come within max(8, 0.1%) of it.
    /// </summary>
    public (int[] Tokens, int DocumentChars) ChatPrompt(Case c, ModelHandle m)
    {
        if (!c.HasDocument) return (m.Render(c.Messages, c.Think), 0);
        string[] paragraphs = Paragraphs(c.Corpus);
        var source = new StringBuilder();
        for (int i = c.StartParagraph; source.Length < (long)c.FitTokens * 8; i++)
        {
            if (source.Length > 0) source.Append("\n\n");
            source.Append(paragraphs[i % paragraphs.Length]);
        }
        string haystack = source.ToString();
        foreach (Insert insert in c.Inserts)
            if (haystack.Contains(insert.Text.Trim(), StringComparison.Ordinal))
                throw new InvalidDataException($"{c.Id}: an insert occurs in the haystack itself");

        int[] Render(int n)
        {
            var doc = new StringBuilder();
            int at = 0;
            foreach (Insert insert in c.Inserts)
            {
                int cut = (int)(n * insert.Depth);
                doc.Append(haystack, at, cut - at).Append(insert.Text);
                at = cut;
            }
            doc.Append(haystack, at, n - at);
            string d = doc.ToString();
            return m.Render(c.Messages.Select(x => (x.Role, x.Content.Replace(Case.DocumentSlot, d, StringComparison.Ordinal))).ToList(), c.Think);
        }

        int target = c.FitTokens, tolerance = Math.Max(8, target / 1000);
        int[] empty = Render(0), full = Render(haystack.Length);
        if (empty.Length > target) throw new InvalidDataException($"{c.Id}: {empty.Length} tokens without a document, more than fit_tokens {target}");
        if (full.Length < target) throw new InvalidDataException($"{c.Id}: the haystack gives only {full.Length} tokens, fit_tokens is {target}");
        if (full.Length == target) return (full, haystack.Length);
        // Proportional first guess, then a bracket [lo: fits, hi: does not] around it, narrowed by bisection until lo is within
        // tolerance (a handful of renders even for 128K-token prompts).
        int guess = (int)((long)haystack.Length * (target - empty.Length) / Math.Max(1, full.Length - empty.Length));
        int lo = 0, hi = haystack.Length;
        int[] best = empty;
        for (int span = Math.Max(64, guess / 64); ; span *= 2)
        {
            int a = Math.Max(0, guess - span);
            int[] ta = Render(a);
            if (ta.Length <= target) { lo = a; best = ta; break; }
            hi = Math.Min(hi, a);
        }
        for (int span = Math.Max(64, guess / 64); ; span *= 2)
        {
            int b = Math.Min(hi, Math.Max(lo, guess) + span);
            if (b >= hi) break;
            int[] tb = Render(b);
            if (tb.Length > target) { hi = b; break; }
            lo = b; best = tb;
        }
        while (target - best.Length > tolerance && hi - lo > 1)
        {
            int mid = lo + (hi - lo) / 2;
            int[] t = Render(mid);
            if (t.Length <= target) { lo = mid; best = t; }
            else hi = mid;
        }
        if (target - best.Length > tolerance)
            throw new InvalidDataException($"{c.Id}: could not fit the prompt to {target} tokens (best {best.Length})");
        return (best, lo);
    }
}
