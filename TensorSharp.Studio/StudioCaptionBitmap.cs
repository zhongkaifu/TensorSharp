// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;

namespace TensorSharp.Studio;

/// <summary>Small independently defined bitmap alphabet used when no external font renderer is installed.
/// Captions are rendered identically in preview and export. Basic Latin is deliberately the supported set;
/// unsupported characters are rejected instead of disappearing from an export.</summary>
internal static class StudioCaptionBitmap
{
    static readonly Dictionary<char, string> Glyphs = new()
    {
        ['A']="01110/10001/10001/11111/10001/10001/10001", ['B']="11110/10001/10001/11110/10001/10001/11110",
        ['C']="01111/10000/10000/10000/10000/10000/01111", ['D']="11110/10001/10001/10001/10001/10001/11110",
        ['E']="11111/10000/10000/11110/10000/10000/11111", ['F']="11111/10000/10000/11110/10000/10000/10000",
        ['G']="01111/10000/10000/10111/10001/10001/01111", ['H']="10001/10001/10001/11111/10001/10001/10001",
        ['I']="11111/00100/00100/00100/00100/00100/11111", ['J']="00111/00010/00010/00010/10010/10010/01100",
        ['K']="10001/10010/10100/11000/10100/10010/10001", ['L']="10000/10000/10000/10000/10000/10000/11111",
        ['M']="10001/11011/10101/10101/10001/10001/10001", ['N']="10001/11001/10101/10011/10001/10001/10001",
        ['O']="01110/10001/10001/10001/10001/10001/01110", ['P']="11110/10001/10001/11110/10000/10000/10000",
        ['Q']="01110/10001/10001/10001/10101/10010/01101", ['R']="11110/10001/10001/11110/10100/10010/10001",
        ['S']="01111/10000/10000/01110/00001/00001/11110", ['T']="11111/00100/00100/00100/00100/00100/00100",
        ['U']="10001/10001/10001/10001/10001/10001/01110", ['V']="10001/10001/10001/10001/10001/01010/00100",
        ['W']="10001/10001/10001/10101/10101/10101/01010", ['X']="10001/10001/01010/00100/01010/10001/10001",
        ['Y']="10001/10001/01010/00100/00100/00100/00100", ['Z']="11111/00001/00010/00100/01000/10000/11111",
        ['0']="01110/10001/10011/10101/11001/10001/01110", ['1']="00100/01100/00100/00100/00100/00100/01110",
        ['2']="01110/10001/00001/00010/00100/01000/11111", ['3']="11110/00001/00001/01110/00001/00001/11110",
        ['4']="00010/00110/01010/10010/11111/00010/00010", ['5']="11111/10000/10000/11110/00001/00001/11110",
        ['6']="01110/10000/10000/11110/10001/10001/01110", ['7']="11111/00001/00010/00100/01000/01000/01000",
        ['8']="01110/10001/10001/01110/10001/10001/01110", ['9']="01110/10001/10001/01111/00001/00001/01110",
        [' ' ]="00000/00000/00000/00000/00000/00000/00000", ['.']="00000/00000/00000/00000/00000/00100/00100",
        [',']="00000/00000/00000/00000/00100/00100/01000", ['!']="00100/00100/00100/00100/00100/00000/00100",
        ['?']="01110/10001/00001/00010/00100/00000/00100", [':']="00000/00100/00100/00000/00100/00100/00000",
        [';']="00000/00100/00100/00000/00100/00100/01000", ['-']="00000/00000/00000/11111/00000/00000/00000",
        ['_']="00000/00000/00000/00000/00000/00000/11111", ['\'']="00100/00100/01000/00000/00000/00000/00000",
        ['"']="01010/01010/01010/00000/00000/00000/00000", ['(']="00010/00100/01000/01000/01000/00100/00010",
        [')']="01000/00100/00010/00010/00010/00100/01000", ['/']="00001/00010/00010/00100/01000/01000/10000",
        ['+']="00000/00100/00100/11111/00100/00100/00000", ['=']="00000/00000/11111/00000/11111/00000/00000",
        ['&']="01100/10010/10100/01000/10101/10010/01101", ['%']="11001/11010/00100/01000/10110/00110/00000",
        ['[']="01110/01000/01000/01000/01000/01000/01110", [']']="01110/00010/00010/00010/00010/00010/01110",
        ['#']="01010/01010/11111/01010/11111/01010/01010", ['*']="00000/10101/01110/11111/01110/10101/00000",
        ['@']="01110/10001/10111/10101/10111/10000/01111", ['$']="00100/01111/10100/01110/00101/11110/00100",
        ['<']="00010/00100/01000/10000/01000/00100/00010", ['>']="01000/00100/00010/00001/00010/00100/01000",
        ['\\']="10000/01000/01000/00100/00010/00010/00001", ['|']="00100/00100/00100/00100/00100/00100/00100",
        ['^']="00100/01010/10001/00000/00000/00000/00000", ['~']="00000/00000/01001/10110/00000/00000/00000",
        ['`']="01000/00100/00010/00000/00000/00000/00000", ['{']="00010/00100/00100/01000/00100/00100/00010",
        ['}']="01000/00100/00100/00010/00100/00100/01000"
    };
    internal static void Validate(string text)
    {
        if (text.Any(c => c is not ('\n' or '\r' or '\t') && !Glyphs.ContainsKey(char.ToUpperInvariant(c))))
            throw new StudioException("caption_characters", "The installed caption renderer supports basic Latin letters, digits, and ASCII punctuation only. Unicode typography is not yet supported.");
    }
    public static void Write(string path, string text, int width, int height)
    {
        Validate(text);
        int scale = Math.Max(1, width / 320); int charWidth = 6 * scale; int lineHeight = 10 * scale;
        int columns = Math.Max(1, (width - 20 * scale) / charWidth);
        var lines = new List<string>();
        foreach (var raw in text.Replace("\r", "").Replace('\t', ' ').Split('\n'))
        {
            var remaining = raw;
            while (remaining.Length > columns)
            {
                var cut = remaining.LastIndexOf(' ', columns - 1, columns);
                if (cut < columns / 2) cut = columns;
                lines.Add(remaining[..cut]); remaining = remaining[cut..].TrimStart();
            }
            lines.Add(remaining);
        }
        if (lines.Count * lineHeight > height * .6) throw new StudioException("caption_too_long", "Caption cannot fit the output frame; shorten or split the caption.");
        var pixels = new byte[checked(width * height * 4)];
        int top = Math.Max(0, height - lines.Count * lineHeight - 20 * scale);
        void Pixel(int x, int y, byte r, byte g, byte b, byte a)
        {
            if (x < 0 || y < 0 || x >= width || y >= height) return;
            int offset = (y * width + x) * 4; pixels[offset] = r; pixels[offset + 1] = g; pixels[offset + 2] = b; pixels[offset + 3] = a;
        }
        for (int l = 0; l < lines.Count; l++)
        {
            int left = (width - lines[l].Length * charWidth) / 2;
            for (int y = top + l * lineHeight - scale; y < top + l * lineHeight + 8 * scale; y++)
                for (int x = left - 2 * scale; x < width - left + scale; x++) Pixel(x, y, 0, 0, 0, 190);
            for (int index = 0; index < lines[l].Length; index++)
            {
                var rows = Glyphs[char.ToUpperInvariant(lines[l][index])].Split('/');
                for (int y = 0; y < 7; y++) for (int x = 0; x < 5; x++) if (rows[y][x] == '1')
                    for (int sy = 0; sy < scale; sy++) for (int sx = 0; sx < scale; sx++) Pixel(left + index * charWidth + x * scale + sx, top + l * lineHeight + y * scale + sy, 255, 255, 255, 255);
            }
        }
        using var file = File.Create(path); file.Write(new byte[] { 137, 80, 78, 71, 13, 10, 26, 10 });
        var header = new byte[13]; BinaryPrimitives.WriteInt32BigEndian(header, width); BinaryPrimitives.WriteInt32BigEndian(header.AsSpan(4), height); header[8] = 8; header[9] = 6; Chunk(file, "IHDR", header);
        using var compressed = new MemoryStream();
        using (var z = new ZLibStream(compressed, CompressionLevel.Fastest, true))
            for (int y = 0; y < height; y++) { z.WriteByte(0); z.Write(pixels, y * width * 4, width * 4); }
        Chunk(file, "IDAT", compressed.ToArray()); Chunk(file, "IEND", []);
    }
    static void Chunk(Stream output, string type, byte[] data)
    {
        Span<byte> number = stackalloc byte[4]; BinaryPrimitives.WriteInt32BigEndian(number, data.Length); output.Write(number);
        var name = Encoding.ASCII.GetBytes(type); output.Write(name); output.Write(data);
        uint crc = 0xffffffff;
        foreach (var b in name.Concat(data))
        {
            crc ^= b;
            for (int i = 0; i < 8; i++) crc = (crc >> 1) ^ ((crc & 1) != 0 ? 0xedb88320u : 0);
        }
        BinaryPrimitives.WriteUInt32BigEndian(number, crc ^ 0xffffffff); output.Write(number);
    }
}
