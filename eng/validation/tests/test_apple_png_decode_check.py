"""The PNG variants and the comparison of the Apple-provider decode check (no Swift needed)."""
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

import numpy as np
from PIL import Image


MODULE = Path(__file__).resolve().parents[1] / "apple-png-decode-check.py"
SPEC = importlib.util.spec_from_file_location("apple_png_decode_check", MODULE)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class ApplePngDecodeCheckTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.out = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    @unittest.skipUnless(CHECK.DISPLAY_P3.exists(), "needs macOS's Display P3 profile")
    def test_each_variant_stores_the_picture_with_only_its_own_chunks(self):
        sample = self.out / "sample.png"
        CHECK.write_png(sample, CHECK.picture(8, 4), [("cHRM", b"\x00" * 32), ("bKGD", b"\x00\x00\x00\x00\x00\x00")])
        files = CHECK.variants(self.out, sample)
        names = sorted(f.name for f in files)
        self.assertEqual(len(names), 14)
        self.assertIn("rgb-imagemagick-chunks.png", names)
        expected = CHECK.picture()
        for path in files:
            stored = np.asarray(Image.open(path).convert("RGBA"))
            self.assertTrue((stored[..., :3] == expected).all(), path.name)
            self.assertTrue((stored[..., 3] == 255).all(), path.name)
        kinds = {f.name: [k for k, _ in CHECK.chunks(f) if k not in ("IDAT", "IEND")] for f in files}
        self.assertEqual(kinds["rgb-plain.png"], ["IHDR"])
        self.assertEqual(kinds["rgba-opaque-display-p3.png"], ["IHDR", "iCCP"])
        self.assertEqual(kinds["rgb-imagemagick-chunks.png"], ["IHDR", "cHRM", "bKGD"])
        header = dict(CHECK.chunks(self.out / "rgba-opaque-plain.png"))["IHDR"]
        self.assertEqual(struct.unpack(">IIBB", header[:10]), (96, 64, 8, 6))
        profile = dict(CHECK.chunks(self.out / "rgb-display-p3.png"))["iCCP"]
        self.assertEqual(zlib.decompress(profile[len(b"ICC Profile") + 2:]), CHECK.DISPLAY_P3.read_bytes())

    def test_the_comparison_counts_colour_values_the_apple_decode_changed(self):
        path = self.out / "rgb-plain.png"
        pixels = CHECK.picture(4, 2)
        CHECK.write_png(path, pixels, [])
        apple = np.concatenate([pixels, np.full((2, 4, 1), 255, np.uint8)], axis=-1)
        (self.out / "rgb-plain.png.rgba").write_bytes(apple.tobytes())
        same = CHECK.compare(path, self.out, "coregraphics sRGB")
        self.assertEqual((same["values_differing"], same["max_difference"], same["values"]), (0, 0, 24))

        apple[1, 2, 0] = (int(apple[1, 2, 0]) + 7) % 256
        apple[0, 0, 3] = 0   # only colour is compared: Core Graphics passes alpha through
        (self.out / "rgb-plain.png.rgba").write_bytes(apple.tobytes())
        changed = CHECK.compare(path, self.out, "coregraphics sRGB")
        self.assertEqual(changed["values_differing"], 1)
        self.assertEqual(changed["max_difference"], abs(int(apple[1, 2, 0]) - int(pixels[1, 2, 0])))

        managed = CHECK.compare(path, self.out, "managed")
        self.assertEqual((managed["apple_decoder"], managed["values_differing"]), ("managed", 0))


if __name__ == "__main__":
    unittest.main()
