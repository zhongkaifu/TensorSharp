#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Build the LFM3-Omni end-to-end media set (macOS): photos, a diagram, text in an image, a 2048-row image, speech with
known transcripts (macOS `say`), a narrated slideshow video with a speech soundtrack, plus llama.cpp's own test media.

    python3 eng/lfm3/omni-in/make-media.py --out artifacts/lfm3/omni-in/media --llama ~/work/llama.cpp-lfm3

Writes <out>/manifest.json: every file with its kind and, for speech, the text that was spoken. Needs PIL, ffmpeg
(PATH), `say` and `sips` (macOS)."""
import argparse
import json
import os
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

HARVARD = {
    # IEEE Harvard sentences (public domain), lists 1-4, first five each.
    "samantha": "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background. "
                "It's easy to tell the depth of a well. These days a chicken leg is a rare dish. "
                "Rice is often served in round bowls.",
    "daniel": "The boy was there when the sun rose. A rod is used to catch pink salmon. "
              "The source of the huge river is the clear spring. Kick the ball straight and follow through. "
              "Help the woman get back to her feet.",
    "karen": "A king ruled the state in the early days. The ship was torn apart on the sharp reef. "
             "Sickness kept him home the third week. The wide road shimmered in the hot sun. "
             "The lazy cow lay in the cool grass.",
    "rishi": "The fish twisted and turned on the bent hook. Press the pants and sew a button on the vest. "
             "The swan dive was far short of perfect. The beauty of the view stunned the young boy. "
             "Two blue fish swam in the tank.",
}
STORY = ("Last Tuesday, Maria took the eight fifteen train to Boston to visit her grandmother. She brought a basket "
         "of apples and a blue scarf. The train was delayed by forty minutes because of heavy snow, so she arrived "
         "just before lunch.")
NARRATION = ["First, mix two cups of flour with one cup of warm water.",
             "Then knead the dough for five minutes until it is smooth.",
             "Finally, bake the bread for twenty minutes at two hundred degrees."]
SLIDES = [("Step 1", "Mix flour and water", (220, 60, 60), "circle"),
          ("Step 2", "Knead the dough", (60, 170, 80), "square"),
          ("Step 3", "Bake the bread", (60, 90, 210), "triangle")]


def run(*cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def font(size):
    for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
              "/Library/Fonts/Arial.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def diagram(path):
    img = Image.new("RGB", (900, 620), "white")
    d = ImageDraw.Draw(img)
    f = font(26)
    boxes = [("Start", 350, 30), ("Load data", 350, 150), ("Train model", 350, 270), ("Accuracy > 90%?", 330, 390),
             ("Deploy", 640, 510), ("Tune parameters", 60, 510)]
    pos = {}
    for text, x, y in boxes:
        w = 260 if "?" in text else 220
        if "?" in text:
            d.polygon([(x + w // 2, y - 10), (x + w + 20, y + 35), (x + w // 2, y + 80), (x - 20, y + 35)], outline="black", width=3, fill=(255, 245, 200))
        else:
            d.rectangle([x, y, x + w, y + 70], outline="black", width=3, fill=(220, 235, 255))
        d.text((x + 18, y + 22), text, fill="black", font=f)
        pos[text] = (x + w // 2, y, y + 70)
    def arrow(a, b, label=None):
        (x0, _, y0), (x1, y1, _) = pos[a], pos[b]
        d.line([x0, y0, x1, y1], fill="black", width=3)
        d.polygon([(x1, y1), (x1 - 8, y1 - 14), (x1 + 8, y1 - 14)], fill="black")
        if label:
            d.text(((x0 + x1) // 2 + 8, (y0 + y1) // 2 - 10), label, fill=(150, 0, 0), font=font(22))
    arrow("Start", "Load data")
    arrow("Load data", "Train model")
    arrow("Train model", "Accuracy > 90%?")
    arrow("Accuracy > 90%?", "Deploy", "yes")
    arrow("Accuracy > 90%?", "Tune parameters", "no")
    d.line([170, 510, 170, 305, 350, 305], fill=(120, 120, 120), width=3)
    img.save(path)


def receipt(path):
    img = Image.new("RGB", (560, 760), (250, 250, 245))
    d = ImageDraw.Draw(img)
    big, f = font(40), font(28)
    d.text((120, 30), "GREEN LEAF CAFE", fill="black", font=big)
    d.text((150, 90), "42 Harbor Street", fill="black", font=f)
    d.text((40, 150), "Date: 2026-03-14   Table 7", fill="black", font=f)
    y = 220
    for item, price in [("Cappuccino", "4.50"), ("Avocado toast", "11.25"), ("Orange juice", "5.00"), ("Blueberry muffin", "3.75")]:
        d.text((40, y), item, fill="black", font=f)
        d.text((420, y), price, fill="black", font=f)
        y += 50
    d.line([40, y + 10, 520, y + 10], fill="black", width=2)
    d.text((40, y + 30), "Subtotal", fill="black", font=f)
    d.text((420, y + 30), "24.50", fill="black", font=f)
    d.text((40, y + 80), "Tax 8%", fill="black", font=f)
    d.text((420, y + 80), "1.96", fill="black", font=f)
    d.text((40, y + 140), "TOTAL", fill="black", font=big)
    d.text((400, y + 140), "$26.46", fill="black", font=big)
    d.text((110, y + 230), "Thank you for visiting!", fill="black", font=f)
    img.save(path)


def say(text, voice, out_wav, tmp):
    aiff = os.path.join(tmp, os.path.basename(out_wav) + ".aiff")
    run("say", "-v", voice, "-o", aiff, text)
    run("ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", aiff, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out_wav)


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         check=True, capture_output=True, text=True).stdout
    return float(out.strip())


def slideshow(out_mp4, tmp):
    # One slide per narration sentence, each shown for its sentence's spoken length.
    parts, concat = [], os.path.join(tmp, "slides.txt")
    with open(concat, "w") as c:
        for i, ((title, text, colour, shape), line) in enumerate(zip(SLIDES, NARRATION)):
            png = os.path.join(tmp, f"slide{i}.png")
            img = Image.new("RGB", (960, 540), (245, 245, 245))
            d = ImageDraw.Draw(img)
            d.text((60, 40), title, fill="black", font=font(64))
            d.text((60, 140), text, fill="black", font=font(48))
            if shape == "circle":
                d.ellipse([620, 240, 860, 480], fill=colour)
            elif shape == "square":
                d.rectangle([620, 240, 860, 480], fill=colour)
            else:
                d.polygon([(740, 230), (870, 480), (610, 480)], fill=colour)
            img.save(png)
            wav = os.path.join(tmp, f"narr{i}.wav")
            say(line, "Samantha", wav, tmp)
            dur = duration(wav) + 0.4
            seg = os.path.join(tmp, f"seg{i}.mp4")
            run("ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-loop", "1", "-framerate", "25", "-t", f"{dur:.3f}", "-i", png,
                "-i", wav, "-af", "apad", "-t", f"{dur:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25",
                "-c:a", "aac", "-ar", "44100", "-ac", "2", seg)
            c.write(f"file '{seg}'\n")
    run("ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", concat, "-c", "copy", out_mp4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--llama", required=True, help="llama.cpp checkout (tools/mtmd/test-1.jpeg, test-2.mp3, test-3.mp4)")
    ap.add_argument("--signal", default=None, help="a short video without sound (InferenceWeb.Tests/Fixtures/JevAttachments/signal.mp4)")
    ap.add_argument("--ticket", default=None, help="a speech WAV with known text (InferenceWeb.Tests/Fixtures/JevAttachments/ticket.wav)")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    tmp = os.path.join(out, "tmp")
    for sub in ("images", "audio", "video", "tmp"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    m = {"images": {}, "audio": {}, "video": {}}
    mtmd = os.path.join(os.path.expanduser(a.llama), "tools", "mtmd")

    # images
    for name in ("Zebra", "Parrot", "Owl"):
        dst = os.path.join(out, "images", f"photo_{name.lower()}.jpg")
        run("sips", "-s", "format", "jpeg", "-s", "formatOptions", "92", f"/Library/User Pictures/Animals/{name}.heic", "--out", dst)
        m["images"][f"photo_{name.lower()}"] = {"path": dst, "kind": "photo"}
    shutil.copy(os.path.join(mtmd, "test-1.jpeg"), os.path.join(out, "images", "newspaper.jpg"))
    m["images"]["newspaper"] = {"path": os.path.join(out, "images", "newspaper.jpg"), "kind": "text-in-photo"}
    diagram(os.path.join(out, "images", "diagram.png"))
    m["images"]["diagram"] = {"path": os.path.join(out, "images", "diagram.png"), "kind": "diagram"}
    receipt(os.path.join(out, "images", "receipt.png"))
    m["images"]["receipt"] = {"path": os.path.join(out, "images", "receipt.png"), "kind": "text-in-image"}
    shutil.copy("/System/Library/Wallpapers/.default/DefaultAerial.jpg", os.path.join(out, "images", "aerial_large.jpg"))
    m["images"]["aerial_large"] = {"path": os.path.join(out, "images", "aerial_large.jpg"), "kind": "large (2048 rows)"}

    # audio with known text
    for voice, text in HARVARD.items():
        dst = os.path.join(out, "audio", f"harvard_{voice}.wav")
        say(text, voice.capitalize(), dst, tmp)
        m["audio"][f"harvard_{voice}"] = {"path": dst, "text": text, "voice": voice}
    dst = os.path.join(out, "audio", "story_moira.wav")
    say(STORY, "Moira", dst, tmp)
    m["audio"]["story_moira"] = {"path": dst, "text": STORY, "voice": "moira"}
    if a.ticket:
        shutil.copy(a.ticket, os.path.join(out, "audio", "ticket.wav"))
        m["audio"]["ticket"] = {"path": os.path.join(out, "audio", "ticket.wav"),
                                "text": "The service is down with a software error. Fix it immediately. I am extremely angry."}
    shutil.copy(os.path.join(mtmd, "test-2.mp3"), os.path.join(out, "audio", "test2.mp3"))
    m["audio"]["test2"] = {"path": os.path.join(out, "audio", "test2.mp3"), "text": None}
    # ~30 s of speech: the four Harvard clips back to back.
    lst = os.path.join(tmp, "long.txt")
    with open(lst, "w") as f:
        for voice in HARVARD:
            f.write(f"file '{os.path.join(out, 'audio', f'harvard_{voice}.wav')}'\n")
    dst = os.path.join(out, "audio", "long_harvard.wav")
    run("ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", dst)
    m["audio"]["long_harvard"] = {"path": dst, "text": " ".join(HARVARD.values())}

    # video
    shutil.copy(os.path.join(mtmd, "test-3.mp4"), os.path.join(out, "video", "test3.mp4"))
    m["video"]["test3"] = {"path": os.path.join(out, "video", "test3.mp4"), "sound": True}
    slideshow(os.path.join(out, "video", "recipe_narrated.mp4"), tmp)
    m["video"]["recipe_narrated"] = {"path": os.path.join(out, "video", "recipe_narrated.mp4"), "sound": True,
                                     "narration": " ".join(NARRATION)}
    if a.signal:
        shutil.copy(a.signal, os.path.join(out, "video", "signal.mp4"))
        m["video"]["signal"] = {"path": os.path.join(out, "video", "signal.mp4"), "sound": False}

    for group in m.values():
        for v in group.values():
            if v["path"].endswith((".wav", ".mp3", ".mp4")):
                v["seconds"] = round(duration(v["path"]), 3)
    shutil.rmtree(tmp, ignore_errors=True)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(m, f, indent=1)
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    sys.exit(main())
