"""Bounded Qwen-Image HTTP/PNG smoke checks, not semantic image qualification.

The caller owns the server, verifies its source/model/backend identity, and supplies
the redirect-refusing loopback HTTP transport. No model downloads/builds occur here.
"""
import hashlib
import json
import math
from pathlib import Path
import struct
import urllib.error
import urllib.parse
import zlib

PNG_LIMIT = 2 * 1024 * 1024
SIZE = 128


class UnsupportedPngFormat(ValueError):
    """A valid encoding shape that this bounded decoder cannot qualify."""

    def __init__(self, metadata):
        self.metadata = metadata
        super().__init__("PNG encoding is not qualified by this decoder: " + json.dumps(metadata, sort_keys=True))


def availability(context):
    options = context.get("model_options", {})
    for option in ("qwen-image-vae", "qwen-image-vl"):
        if not options.get(option):
            return False, "Qwen image generation requires companion " + option
        if not context.get("preparing_model") and not Path(options[option]).is_file():
            return False, "verified Qwen image companion is absent: " + option
    variant = options.get("qwen-image-variant", "base")
    if variant not in {"base", "turbo"}:
        return False, "image smoke supports source-defined base/turbo schedules only"
    return True, "128px Qwen image smoke prerequisites present"


def local_artifact_route(url, base):
    """Never contact a different origin or reinterpret a server URL as local disk."""
    if not isinstance(url, str) or not url or any(ord(c) < 32 for c in url) or "\\" in url:
        raise ValueError("invalid generated artifact URL")
    value, origin = urllib.parse.urlsplit(url), urllib.parse.urlsplit(base)
    if value.fragment or value.username is not None or value.password is not None:
        raise ValueError("artifact URL credentials/fragments are refused")
    if value.scheme or value.netloc:
        if (value.scheme, value.netloc) != (origin.scheme, origin.netloc):
            raise ValueError("generated artifact URL is outside owned loopback origin")
    if not value.path.startswith("/") or value.path.startswith("//"):
        raise ValueError("generated artifact route must be root-relative")
    return value.path + ("?" + value.query if value.query else "")


def _paeth(a, b, c):
    estimate = a + b - c
    distances = (abs(estimate - a), abs(estimate - b), abs(estimate - c))
    return a if distances[0] <= distances[1] and distances[0] <= distances[2] else b if distances[1] <= distances[2] else c


def png_metadata(data, width=SIZE, height=SIZE):
    """Strict bounded noninterlaced PNG decoder, including all five PNG filters.

    Accept RGB/RGBA8, gray/gray-alpha8 and packed gray/palette encodings that
    desktop encoders can choose. Normalize to RGBA8 for comparisons. Legal
    16-bit/Adam7 encodings remain explicitly untested instead of engine failures.
    Finite byte pixels do not qualify intermediate tensors or semantic quality.
    """
    if not isinstance(data, bytes) or len(data) > PNG_LIMIT or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("invalid or oversized PNG signature")
    offset, compressed, header, ended, idat_ended, seen_idat = 8, bytearray(), None, False, False, False
    palette, transparency = None, None
    while offset < len(data):
        if len(data) - offset < 12:
            raise ValueError("truncated PNG chunk")
        length = struct.unpack_from(">I", data, offset)[0]
        if length > len(data) - offset - 12:
            raise ValueError("truncated PNG chunk payload")
        kind = data[offset + 4:offset + 8]
        if any(not (65 <= value <= 90 or 97 <= value <= 122) for value in kind) or not 65 <= kind[2] <= 90:
            raise ValueError("invalid PNG chunk type")
        content = data[offset + 8:offset + 8 + length]
        crc = struct.unpack_from(">I", data, offset + 8 + length)[0]
        if zlib.crc32(kind + content) & 0xffffffff != crc:
            raise ValueError("PNG CRC mismatch")
        if header is None and kind != b"IHDR":
            raise ValueError("PNG must start with IHDR")
        if kind == b"IHDR":
            if header is not None or length != 13:
                raise ValueError("invalid or repeated PNG IHDR")
            header = struct.unpack(">IIBBBBB", content)
            actual_width, actual_height, depth, color, compression, filtering, interlace = header
            legal_depths = {0: {1, 2, 4, 8, 16}, 2: {8, 16}, 3: {1, 2, 4, 8}, 4: {8, 16}, 6: {8, 16}}
            if (actual_width, actual_height) != (width, height):
                raise ValueError("PNG dimensions differ from requested dimensions")
            if color not in legal_depths or depth not in legal_depths[color] or compression != 0 or filtering != 0 or interlace not in {0, 1}:
                raise ValueError("invalid PNG encoding parameters")
        elif kind == b"PLTE":
            if palette is not None or seen_idat or color in {0, 4} or not 0 < length <= 768 or length % 3:
                raise ValueError("invalid PNG palette")
            if color == 3 and length // 3 > (1 << depth):
                raise ValueError("PNG palette exceeds indexed bit depth")
            palette = content
        elif kind == b"tRNS":
            if transparency is not None or seen_idat or color in {4, 6}:
                raise ValueError("invalid PNG transparency chunk")
            if (color == 0 and length != 2) or (color == 2 and length != 6) or (color == 3 and (palette is None or not 0 < length <= len(palette) // 3)):
                raise ValueError("invalid PNG transparency entries")
            transparency = content
        elif kind == b"IDAT":
            if idat_ended:
                raise ValueError("noncontiguous PNG IDAT chunks")
            seen_idat = True
            compressed.extend(content)
        elif kind == b"IEND":
            if length or not compressed:
                raise ValueError("invalid PNG IEND or absent pixels")
            ended = True
            offset += 12
            break
        else:
            if seen_idat:
                idat_ended = True
            if kind[:1].isupper() and kind != b"PLTE":
                raise ValueError("unsupported critical PNG chunk")
        offset += 12 + length
    if not ended or offset != len(data):
        raise ValueError("PNG IEND absent or trailing data present")
    if color == 3 and palette is None:
        raise ValueError("indexed PNG has no palette")
    encoding = {"width": width, "height": height, "bit_depth": depth, "color_type": color, "interlace": interlace}
    if depth == 16 or interlace:
        raise UnsupportedPngFormat(encoding)
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
    stride = (width * channels * depth + 7) // 8
    bpp = max(1, (channels * depth + 7) // 8)
    expected = height * (stride + 1)
    inflater = zlib.decompressobj()
    try:
        inflated = inflater.decompress(bytes(compressed), expected + 1)
    except zlib.error as error:
        raise ValueError("invalid PNG zlib stream") from error
    if len(inflated) != expected or not inflater.eof or inflater.unused_data or inflater.unconsumed_tail:
        raise ValueError("PNG pixel stream is truncated, oversized or has trailing data")
    previous, pixels = bytearray(stride), bytearray()
    transparent_samples = None
    if transparency is not None and color in {0, 2}:
        transparent_samples = struct.unpack(">" + "H" * channels, transparency)
        if any(sample >= (1 << depth) for sample in transparent_samples):
            raise ValueError("PNG transparent sample exceeds bit depth")
    for y in range(height):
        start = y * (stride + 1)
        kind = inflated[start]
        if kind not in range(5):
            raise ValueError("invalid PNG row filter")
        row = bytearray(inflated[start + 1:start + stride + 1])
        for i in range(stride):
            left, up, diagonal = (row[i - bpp] if i >= bpp else 0), previous[i], (previous[i - bpp] if i >= bpp else 0)
            predict = 0 if kind == 0 else left if kind == 1 else up if kind == 2 else (left + up) // 2 if kind == 3 else _paeth(left, up, diagonal)
            row[i] = (row[i] + predict) & 255
        for x in range(width):
            if depth < 8:
                sample = (row[(x * depth) // 8] >> (8 - depth - (x * depth) % 8)) & ((1 << depth) - 1)
                samples = (sample,)
            else:
                samples = tuple(row[x * channels:(x + 1) * channels])
            if color == 3:
                index = samples[0]
                if index >= len(palette) // 3:
                    raise ValueError("PNG pixel index is outside palette")
                rgb = palette[index * 3:index * 3 + 3]
                alpha = transparency[index] if transparency is not None and index < len(transparency) else 255
            elif color in {0, 4}:
                gray = samples[0] * 255 // ((1 << depth) - 1)
                rgb = (gray, gray, gray)
                alpha = samples[1] if color == 4 else 0 if samples == transparent_samples else 255
            else:
                rgb = samples[:3]
                alpha = samples[3] if color == 6 else 0 if samples == transparent_samples else 255
            pixels.extend((*rgb, alpha))
        previous = row
    first = pixels[:4]
    varied = any(pixels[i:i + 4] != first for i in range(4, len(pixels), 4))
    visible = any(pixels[i] > 0 for i in range(3, len(pixels), 4))
    if not varied or not visible:
        raise ValueError("PNG output is spatially constant or fully transparent")
    return {**encoding, "decoded_format": "RGBA8",
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "pixel_sha256": hashlib.sha256(pixels).hexdigest(), "spatially_varied": varied,
            "has_visible_pixels": visible, "alpha_min": min(pixels[3::4]), "alpha_max": max(pixels[3::4])}


def run_image_smoke(http, model, context, result):
    """Run bounded transport/output checks through the caller's owned HTTP client."""
    from .scenarios import ScenarioFailure, ScenarioUntested, _require
    options = context.get("model_options", {})
    steps = 8 if options.get("qwen-image-variant") == "turbo" else 2
    payload = {"prompt": "A red ceramic cube on a white table.", "width": SIZE, "height": SIZE,
               "steps": steps, "cfg": 1, "seed": 42, "negativePrompt": " "}
    result["inputs"]["image_generation"] = {"hosted_model": model, **payload}
    scope = {"resolution": "128x128", "semantic_quality": "untested; no approved perceptual/reference fixture",
             "internal_tensor_finiteness": "untested; decoded pixels are finite unsigned bytes",
             "seeded_repeat_rule": "observe file/pixel SHA equality; no universal bitwise backend guarantee asserted",
             "editing_streaming_cancellation_lora": "untested by this smoke adapter"}
    result["evidence"]["image_scope"] = scope
    negatives = ((dict(payload, width=SIZE + 1), "width and height"),
                 (dict(payload, prompt=""), "nonempty prompt"),
                 (dict(payload, prompt=17), "prompt must be a string"),
                 (dict(payload, width="128"), "Invalid image-generation parameters"))
    for body, expected_message in negatives:
        try:
            http.json("/api/image-generate", body)
        except urllib.error.HTTPError as error:
            records = result.get("evidence", {}).get("http_errors", [])
            try:
                response = json.loads(records[-1]["body"]) if records else {}
            except (ValueError, KeyError, TypeError):
                response = {}
            _require(result, "malformed image request rejected with diagnostic", error.code == 400 and
                     isinstance(response, dict) and isinstance(response.get("error"), str) and expected_message in response["error"],
                     {"status": 400, "error_contains": expected_message}, {"status": error.code, "body": response})
        else:
            raise ScenarioFailure("malformed image-generation request was accepted")
    images = []
    result["evidence"]["generated_images"] = images
    for iteration in range(2):
        value = http.json("/api/image-generate", payload)
        _require(result, "image result metadata", isinstance(value, dict) and value.get("ok") is True and
                 value.get("width") == SIZE and value.get("height") == SIZE,
                 {"ok": True, "width": SIZE, "height": SIZE}, value)
        elapsed = value.get("elapsedSeconds")
        _require(result, "finite generation duration", type(elapsed) in {int, float} and math.isfinite(elapsed) and elapsed >= 0,
                 "finite nonnegative seconds", elapsed)
        try:
            route = local_artifact_route(value.get("url"), http.base)
            with http.open(route) as response:
                if response.status != 200 or response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "image/png":
                    raise ValueError("generated artifact is not an HTTP 200 image/png")
                data = response.read(PNG_LIMIT + 1)
            target = Path(context["artifact_dir"]) / ("image-smoke-" + str(iteration + 1) + ".png")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data[:PNG_LIMIT])
            receipt = {"artifact": str(target), "route": route, "captured_bytes": min(len(data), PNG_LIMIT),
                       "capture_truncated": len(data) > PNG_LIMIT, "validation": "incomplete"}
            images.append(receipt)
            info = png_metadata(data)
        except UnsupportedPngFormat as error:
            receipt.update(validation="untested", encoding=error.metadata, reason=str(error))
            raise ScenarioUntested(str(error)) from error
        except (ValueError, TypeError, KeyError, struct.error) as error:
            raise ScenarioFailure(str(error)) from error
        receipt.update(info, validation="passed")
        _require(result, "complete nonconstant visible PNG", True, "128x128 valid PNG decoded to RGBA8", info)
    result["evidence"]["generated_images"] = images
    result["evidence"]["image_seeded_repeat"] = {
        "file_sha_equal": images[0]["sha256"] == images[1]["sha256"],
        "pixel_sha_equal": images[0]["pixel_sha256"] == images[1]["pixel_sha256"],
        "qualified": False, "reason": "observation only; source guarantees seeded noise, not every backend's bitwise final output"}
