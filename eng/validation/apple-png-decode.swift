// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Decodes PNG files the way TensorSharp's Apple media provider does
// (TensorSharp.Models/Media/Apple/AppleMediaProvider.cs, DecodeRgba), so a desktop Mac can tell
// which PNGs the TensorAgent apps read differently from the CLI and server (ImageMagick keeps a
// PNG's stored values). Driven by apple-png-decode-check.py, which compares the result with the
// stored values.
//
//   swift eng/validation/apple-png-decode.swift <out-dir> <file.png>...
//
// For each file prints one line, "<file> managed" or "<file> coregraphics", and for the second
// writes <out-dir>/<file name>.rgba: straight RGBA8, unpadded, as the provider returns it.
// "managed" is the provider's own PNG codec, which returns the stored values: it takes a PNG that
// carries alpha (colour type 4 or 6, or a tRNS chunk) and embeds no colour profile (iCCP).
// Everything else is drawn by Core Graphics into an 8-bit sRGB bitmap context, 1:1, without
// interpolation, as the provider does. EXIF orientation is not applied (the probe's files have none).
import CoreGraphics
import Foundation
import ImageIO

func chunks(_ data: Data) -> [(String, Data)] {
    var result: [(String, Data)] = []
    var offset = 8
    let bytes = [UInt8](data)
    while offset + 8 <= bytes.count {
        let length = Int(bytes[offset]) << 24 | Int(bytes[offset + 1]) << 16 | Int(bytes[offset + 2]) << 8 | Int(bytes[offset + 3])
        let type = String(bytes: bytes[(offset + 4)..<(offset + 8)], encoding: .isoLatin1) ?? ""
        let start = offset + 8
        guard start + length <= bytes.count else { break }
        result.append((type, Data(bytes[start..<(start + length)])))
        if type == "IEND" { break }
        offset = start + length + 4
    }
    return result
}

/// PngCodec.CarriesAlpha && !PngCodec.HasChunkBeforeImageData(file, "iCCP").
func takesManagedCodec(_ data: Data) -> Bool {
    let list = chunks(data)
    guard let header = list.first, header.0 == "IHDR", header.1.count >= 10 else { return false }
    let colorType = header.1[header.1.startIndex + 9]
    var tRNS = false, iCCP = false
    for (type, _) in list {
        if type == "IDAT" || type == "IEND" { break }
        if type == "tRNS" { tRNS = true }
        if type == "iCCP" { iCCP = true }
    }
    return (colorType == 4 || colorType == 6 || tRNS) && !iCCP
}

/// AppleMediaProvider.DrawStraightRgba.
func drawStraightRgba(_ image: CGImage) -> [UInt8] {
    let width = image.width, height = image.height
    let hasAlpha = !(image.alphaInfo == .none || image.alphaInfo == .noneSkipLast || image.alphaInfo == .noneSkipFirst)
    let space = CGColorSpace(name: CGColorSpace.sRGB)!
    let info = hasAlpha ? CGImageAlphaInfo.premultipliedLast.rawValue : CGImageAlphaInfo.noneSkipLast.rawValue
    let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                            space: space, bitmapInfo: info)!
    context.setShouldAntialias(false)
    context.interpolationQuality = .none
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    let rowBytes = context.bytesPerRow
    let pixels = context.data!.assumingMemoryBound(to: UInt8.self)
    var rgba = [UInt8](repeating: 0, count: width * height * 4)
    for y in 0..<height {
        for x in 0..<(width * 4) { rgba[y * width * 4 + x] = pixels[y * rowBytes + x] }
    }
    for i in stride(from: 0, to: rgba.count, by: 4) {
        if !hasAlpha { rgba[i + 3] = 255; continue }
        // StraightAlpha.Unpremultiply: round(c * 255 / a), 0 where a is 0.
        let a = Int(rgba[i + 3])
        for c in 0..<3 { rgba[i + c] = a == 0 ? 0 : UInt8(min(255, (Int(rgba[i + c]) * 255 + a / 2) / a)) }
    }
    return rgba
}

let arguments = CommandLine.arguments
guard arguments.count >= 3 else {
    FileHandle.standardError.write("usage: apple-png-decode.swift <out-dir> <file.png>...\n".data(using: .utf8)!)
    exit(2)
}
let outDir = URL(fileURLWithPath: arguments[1])
for path in arguments.dropFirst(2) {
    let url = URL(fileURLWithPath: path)
    let data = try Data(contentsOf: url)
    if takesManagedCodec(data) {
        print("\(url.lastPathComponent) managed")
        continue
    }
    guard let source = CGImageSourceCreateWithData(data as CFData, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCache: false] as CFDictionary) else {
        print("\(url.lastPathComponent) undecodable")
        continue
    }
    let rgba = drawStraightRgba(image)
    try Data(rgba).write(to: outDir.appendingPathComponent(url.lastPathComponent + ".rgba"))
    print("\(url.lastPathComponent) coregraphics \(image.colorSpace?.name as String? ?? "untagged")")
}
