// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading;
using System.Xml;
using System.Xml.Linq;

namespace TensorSharp.Server.Jev;

/// <summary>Bounded managed OOXML text extraction. No macros, external relationships, network
/// resources or formulas are executed. Archive members are read in place, never unpacked.</summary>
internal static class JevAttachmentOfficeText
{
    internal const long MaxExpandedBytes = 8 * 1024 * 1024;
    private const int MaxEntries = 2048;

    internal static string Extract(string path, string extension, int maxCharacters, CancellationToken cancellationToken)
    {
        using var archive = ZipFile.OpenRead(path);
        if (archive.Entries.Count > MaxEntries || archive.Entries.Any(entry => entry.Length > MaxExpandedBytes) ||
            archive.Entries.Sum(entry => entry.Length) > MaxExpandedBytes)
            throw new InvalidDataException($"Office archive exceeds {MaxEntries} entries or {MaxExpandedBytes} expanded bytes");
        if (archive.Entries.Select(entry => entry.FullName).Distinct(StringComparer.Ordinal).Count() != archive.Entries.Count)
            throw new InvalidDataException("Office archive contains duplicate entries");
        var output = new StringBuilder();
        void Append(string text)
        {
            if (output.Length + text.Length > maxCharacters)
                throw new InvalidDataException($"extracted text exceeds {maxCharacters} characters; split the attachment");
            output.Append(text);
        }
        XDocument Read(ZipArchiveEntry entry)
        {
            cancellationToken.ThrowIfCancellationRequested();
            using var stream = entry.Open();
            using var reader = XmlReader.Create(stream, new XmlReaderSettings
            {
                DtdProcessing = DtdProcessing.Prohibit,
                XmlResolver = null,
                MaxCharactersInDocument = MaxExpandedBytes,
                IgnoreComments = true,
            });
            return XDocument.Load(reader);
        }
        IEnumerable<XElement> Elements(XContainer node, string name) => node.Descendants().Where(e => e.Name.LocalName == name);
        string Text(XContainer node) => string.Concat(node.Descendants().Select(e => e.Name.LocalName switch
        { "t" => e.Value, "br" or "cr" => "\n", "tab" => "\t", _ => "" }));
        Dictionary<string, string> Relationships(string name)
        {
            var entry = archive.GetEntry(name) ?? throw new InvalidDataException("Office document is missing " + name);
            var relationships = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var relation in Elements(Read(entry), "Relationship"))
            {
                string? id = (string?)relation.Attribute("Id"), target = (string?)relation.Attribute("Target");
                if (string.IsNullOrEmpty(id) || string.IsNullOrEmpty(target) || relationships.ContainsKey(id))
                    throw new InvalidDataException("Office document has malformed or duplicate relationships");
                if ((string?)relation.Attribute("TargetMode") == "External") continue;
                relationships.Add(id, target);
            }
            return relationships;
        }
        ZipArchiveEntry RelatedEntry(XElement element, Dictionary<string, string> relationships, string root)
        {
            string? id = element.Attributes().FirstOrDefault(a => a.Name.LocalName == "id" && a.Name.NamespaceName.Length > 0)?.Value;
            if (id == null || !relationships.TryGetValue(id, out string? target))
                throw new InvalidDataException("Office document contains an unresolved relationship");
            string name = target.StartsWith("/", StringComparison.Ordinal) ? target[1..] : root + "/" + target;
            if (name.Contains("..", StringComparison.Ordinal)) throw new InvalidDataException("Office document contains an invalid relationship target");
            return archive.GetEntry(name) ?? throw new InvalidDataException("Office document is missing " + name);
        }

        if (extension == ".docx")
        {
            var main = archive.GetEntry("word/document.xml") ?? throw new InvalidDataException("DOCX is missing word/document.xml");
            var document = Read(main);
            void Paragraphs(XContainer content)
            {
                foreach (var paragraph in Elements(content, "p"))
                {
                    // Text runs concatenate within a word; explicit tabs and breaks separate
                    // table cells and lines without inserting spaces into styled words.
                    foreach (var element in paragraph.Descendants())
                        if (element.Name.LocalName == "t") Append(element.Value);
                        else if (element.Name.LocalName == "tab") Append("\t");
                        else if (element.Name.LocalName is "br" or "cr") Append("\n");
                    Append("\n");
                }
            }
            Paragraphs(document);
            // ZIP membership does not mean a part is displayed: editing a document can
            // leave orphaned headers/footers with obsolete evidence behind in the package.
            var references = document.Descendants().Where(element => element.Name.LocalName is "headerReference" or "footerReference").ToArray();
            if (references.Length > 0)
            {
                var relationships = Relationships("word/_rels/document.xml.rels");
                foreach (var entry in references.Select(reference => RelatedEntry(reference, relationships, "word")).DistinctBy(entry => entry.FullName))
                    Paragraphs(Read(entry));
            }
            foreach (string noteType in new[] { "footnote", "endnote" })
            {
                var noteIds = Elements(document, noteType + "Reference")
                    .Select(note => note.Attributes().FirstOrDefault(attribute => attribute.Name.LocalName == "id")?.Value)
                    .Where(id => id != null).ToHashSet(StringComparer.Ordinal);
                if (noteIds.Count == 0) continue;
                var entry = archive.GetEntry("word/" + noteType + "s.xml") ?? throw new InvalidDataException("DOCX is missing referenced " + noteType + "s");
                foreach (var note in Elements(Read(entry), noteType))
                    if (noteIds.Contains(note.Attributes().FirstOrDefault(attribute => attribute.Name.LocalName == "id")?.Value))
                        Paragraphs(note);
            }
        }
        else if (extension == ".pptx")
        {
            var presentation = archive.GetEntry("ppt/presentation.xml") ?? throw new InvalidDataException("PPTX is missing ppt/presentation.xml");
            var relationships = Relationships("ppt/_rels/presentation.xml.rels");
            var slides = Elements(Read(presentation), "sldId").Select(slide => RelatedEntry(slide, relationships, "ppt")).ToArray();
            if (slides.Length == 0) throw new InvalidDataException("PPTX contains no readable slides");
            int slideNumber = 0;
            foreach (var slide in slides)
            {
                Append($"Slide {++slideNumber}:\n");
                foreach (var paragraph in Elements(Read(slide), "p")) Append(Text(paragraph) + "\n");
            }
        }
        else if (extension == ".xlsx")
        {
            var sharedEntry = archive.GetEntry("xl/sharedStrings.xml");
            string[] shared = sharedEntry == null ? [] : Elements(Read(sharedEntry), "si").Select(Text).ToArray();
            var workbook = archive.GetEntry("xl/workbook.xml") ?? throw new InvalidDataException("XLSX is missing xl/workbook.xml");
            var relationships = Relationships("xl/_rels/workbook.xml.rels");
            foreach (var sheet in Elements(Read(workbook), "sheet"))
            {
                cancellationToken.ThrowIfCancellationRequested();
                // ZIP lookup only, with no filesystem access or external target resolution.
                var entry = RelatedEntry(sheet, relationships, "xl");
                Append("Sheet " + (string?)sheet.Attribute("name") + ":\n");
                foreach (var row in Elements(Read(entry), "row"))
                {
                    foreach (var cell in row.Elements().Where(e => e.Name.LocalName == "c"))
                    {
                        string? type = (string?)cell.Attribute("t");
                        string? value = cell.Elements().FirstOrDefault(e => e.Name.LocalName == "v")?.Value;
                        if (type == "inlineStr") value = Text(cell);
                        else if (type == "s")
                        {
                            if (!int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out int index) || index < 0 || index >= shared.Length)
                                throw new InvalidDataException("XLSX has an invalid shared-string reference");
                            value = shared[index];
                        }
                        else if (type == "b") value = value == "1" ? "true" : "false";
                        if (value == null && cell.Elements().Any(e => e.Name.LocalName == "f")) value = "[formula has no cached value]";
                        if (value != null) Append((string?)cell.Attribute("r") + "=" + value + "\t");
                    }
                    Append("\n");
                }
            }
        }
        else throw new InvalidDataException("unsupported Office format");
        if (string.IsNullOrWhiteSpace(output.ToString())) throw new InvalidDataException("Office document contains no readable text");
        return output.ToString().TrimEnd();
    }

}
