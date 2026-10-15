#!/usr/bin/env python3
"""Generate tiny independent Pillow pixel goldens; never imports TensorSharp.

Use the pinned Pillow version. The checked-in JSON is sufficient for offline
.NET tests; Python/Pillow are needed only to deliberately regenerate it.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import PIL
from PIL import Image, ImageDraw

PILLOW = '12.3.0'
SOURCES = [
    'https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py',
    'https://github.com/python-pillow/Pillow/blob/12.3.0/src/libImaging/Resample.c',
    'https://github.com/python-pillow/Pillow/blob/12.3.0/src/libImaging/AlphaComposite.c',
    'https://github.com/ggml-org/llama.cpp/blob/4ebdf2c74acce30883d8e34b7c70b3eb8146f2fe/tools/mtmd/mtmd-image.cpp',
]


def pattern(width, height, alpha=False):
    image = Image.new('RGBA', (width, height))
    pixels = []
    for y in range(height):
        for x in range(width):
            pixels.append(((47*x + 89*y + 13*x*y) % 256,
                           (211*x + 17*y + 29*x*y + 7) % 256,
                           (67*x + 103*y + 3*x*y + 19) % 256,
                           [0,1,64,127,128,192,254,255][(x + 3*y) % 8] if alpha else 255))
    image.putdata(pixels)
    return image


def checker(width, height):
    image = Image.new('RGBA', (width, height))
    image.putdata([(255 if (x+y) % 2 else 0,
                    255 if (3*x+y) % 4 else 0,
                    (x*71 + y*113) % 256,255) for y in range(height) for x in range(width)])
    return image


def edges():
    image = Image.new('RGBA', (5,7), (0,0,0,255))
    for x,y,color in [(0,0,(255,255,255,255)),(4,0,(255,0,0,255)),
                      (0,6,(0,255,0,255)),(4,6,(0,0,255,255)),(2,3,(255,255,0,255))]:
        image.putpixel((x,y),color)
    return image


def card():
    image = Image.new('RGBA',(32,24),(255,255,255,255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((4,13,27,20), fill=(225,19,31,255))
    # Fixed bitmap ink rather than a font whose rasterization can change.
    glyphs = ['101111001001','111101111101','111001111100','010110010111']
    for index,glyph in enumerate(glyphs):
        for cell,value in enumerate(glyph):
            if value == '1':
                x,y = 5 + index*6 + cell%3, 3 + cell//3
                draw.rectangle((x,y,x,y+1), fill=(0,0,0,255))
    return image


def fixture(name, image, target, automatic=None):
    white = Image.new('RGBA',image.size,(255,255,255,255))
    rgb = Image.alpha_composite(white,image).convert('RGB')
    expected = rgb.resize(target, Image.Resampling.BICUBIC)
    png = io.BytesIO(); image.save(png,format='PNG',compress_level=9)
    raw = image.tobytes(); result = expected.tobytes()
    row = {'Name':name,'SourceWidth':image.width,'SourceHeight':image.height,
           'TargetWidth':target[0],'TargetHeight':target[1],
           'InputRgbaBase64':base64.b64encode(raw).decode(),
           'InputPngBase64':base64.b64encode(png.getvalue()).decode(),
           'ExpectedRgbBase64':base64.b64encode(result).decode(),
           'InputRgbaSha256':hashlib.sha256(raw).hexdigest(),
           'ExpectedRgbSha256':hashlib.sha256(result).hexdigest(),
           'AutomaticResize':automatic is not None}
    if automatic:
        row['MinimumPixels'],row['MaximumPixels'] = automatic
    return row


def generate():
    if PIL.__version__ != PILLOW:
        raise ValueError('Require pinned Pillow ' + PILLOW + ', found ' + PIL.__version__)
    cases = [
        fixture('upscale-pattern',pattern(3,2),(9,6),(54,54)),
        fixture('downscale-checker',checker(8,6),(4,3),(12,12)),
        fixture('mixed-axis-corner-impulses',edges(),(9,3)),
        fixture('single-column-up-down',pattern(1,5),(7,3)),
        fixture('single-row-down-up',pattern(5,1),(3,7)),
        fixture('alpha-rounded-white',pattern(4,3,True),(7,5)),
        fixture('no-resize-opaque',pattern(5,4),(5,4),(1,20)),
        fixture('no-resize-alpha',pattern(4,3,True),(4,3)),
        fixture('card-red-rectangle-black-ink',card(),(20,15),(300,300)),
        fixture('odd-strong-downscale',pattern(11,9),(3,2)),
        fixture('high-frequency-antialias',checker(17,11),(5,3)),
    ]
    return {'Schema':'qwen-byte-bicubic-pillow-reference.v1','PillowVersion':PILLOW,
            'Sources':SOURCES,'Generator':'eng/validation/generate-qwen-bicubic-reference.py',
            'Pipeline':'RGBA -> Pillow.alpha_composite(opaque white, input) -> RGB uint8 -> Pillow.Image.resize(BICUBIC) -> expected RGB uint8; tests check CHW normalization separately.',
            'Scope':'Independent pixel reference only; no model/backend/device or generation-quality qualification.',
            'Cases':cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--check',action='store_true',help='Compare reproducible bytes without overwriting the fixture')
    args = parser.parse_args()
    encoded = (json.dumps(generate(),ensure_ascii=False,indent=2) + '\n').encode()
    if args.check:
        if args.output.read_bytes() != encoded:
            raise ValueError('Checked-in oracle differs from pinned Pillow output')
    else:
        if args.output.exists():
            raise ValueError('Refuse to replace an existing fixture without explicit removal/review')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_bytes(encoded)


if __name__ == '__main__':
    main()
