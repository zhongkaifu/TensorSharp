#!/usr/bin/env python3
"""Check that Qwen-Image-2.1 edits keep their source picture's size, on a real server.

Starts one TensorSharp.Server.Host with Qwen-Image-2.1 (and optionally a LoRA plug-in), makes a
non-square source larger than the edit area from --source, and runs the chains the pages make:

  tensoragent: a selection edit, then a change of its result (TensorAgent's 1 MP targetArea);
  webui:       an edit, then an edit of its result through multipart (the server's default area);

each with keepSourceSize, plus the same change without it for contrast, and optionally a source
smaller than the area with and without it. Every keepSourceSize result must have the source's
exact width and height. Evidence (PNGs, SSE frames, the server log and summary.json) goes to
--output, which belongs under the ignored artifacts/ or docs/validation/. This checks geometry
and records timings; whether each edit followed its instruction is judged by looking at the PNGs.

  python3 eng/validation/qwen-image21-keep-source-size.py --dit .../qwen_image_2.1_Q4_K_M.gguf \\
      --vae ... --text-encoder ... --mmproj ... --source fox.png --output artifacts/keep-source-size
"""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import signal
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request
import uuid

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TENSORAGENT_AREA = 1024 * 1024


def png_size(data):
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise AssertionError("Not a PNG")
    return struct.unpack(">II", data[16:24])


def multipart(fields, files):
    boundary = uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
    for name, filename, data in files:
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
                 "Content-Type: image/png\r\n\r\n").encode() + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def size_arg(text):
    match = re.fullmatch(r"(\d+)x(\d+)", text)
    if not match:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT")
    return int(match.group(1)), int(match.group(2))


def make_source(path, size, out):
    """Centre-crop to the target shape, then Lanczos-resize: a real picture at a known size."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = size
        target, actual = width / height, image.width / image.height
        if actual > target:
            crop = round(image.height * target)
            image = image.crop(((image.width - crop) // 2, 0, (image.width - crop) // 2 + crop, image.height))
        elif actual < target:
            crop = round(image.width / target)
            image = image.crop((0, (image.height - crop) // 2, image.width, (image.height - crop) // 2 + crop))
        image.resize(size, Image.LANCZOS).save(out)
    return out.read_bytes()


def make_mask(size, out):
    """White (edit) over the lowest 35% of the picture, black (keep) elsewhere."""
    width, height = size
    mask = Image.new("L", size, 0)
    mask.paste(255, (0, round(height * 0.65), width, height))
    mask.save(out)
    return out.read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dit", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--lora", type=Path, help="LoRA plug-in weights, loaded at server startup.")
    parser.add_argument("--lora-config", type=Path, help="Its TensorSharp config (strength and sampling recipe).")
    parser.add_argument("--source", type=Path, required=True, help="Any picture; cropped and resized to --size.")
    parser.add_argument("--size", type=size_arg, default=(1600, 1200), help="Source size, larger than 1 MP.")
    parser.add_argument("--small-size", type=size_arg, help="Also edit a source smaller than the area (e.g. 640x480).")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seeds", default="5,7", help="Seeds of the first and second edit of each chain.")
    parser.add_argument("--backend", default="ggml_metal")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5031)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/validation/qwen-image21-keep-source-size")
    parser.add_argument("--skip-contrast", action="store_true", help="Skip the change made without keepSourceSize.")
    args = parser.parse_args()
    first_seed, second_seed = (int(s) for s in args.seeds.split(","))

    executable = ROOT / "TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll"
    if not executable.is_file():
        parser.error("Build TensorSharp.Server.Host before running this harness")
    for path in (args.dit, args.vae, args.text_encoder, args.mmproj, args.source):
        if not path.is_file():
            parser.error(f"Missing file: {path}")
    with socket.socket() as probe:
        if probe.connect_ex((args.host, args.port)) == 0:
            parser.error("Port is already in use; refusing to test or terminate an unrelated server")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = ["dotnet", str(executable), "--model", str(args.dit.resolve()), "--backend", args.backend,
               "--qwen-image-vae", str(args.vae.resolve()), "--qwen-image-vl", str(args.text_encoder.resolve()),
               "--qwen-image-mmproj", str(args.mmproj.resolve()), "--host", args.host, "--port", str(args.port)]
    if args.lora:
        command += ["--lora", str(args.lora.resolve())]
        if args.lora_config:
            command += ["--lora-config", str(args.lora_config.resolve())]
    base = f"http://{args.host}:{args.port}"
    server_log = output / "server.log"
    report = {"purpose": "Edit geometry with keepSourceSize on a real model; quality is judged from the PNGs",
              "command": command, "server_dll_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
              "source": str(args.source), "size": list(args.size), "steps": args.steps,
              "seeds": [first_seed, second_seed], "cases": []}

    def request(path, payload=None, content_type="application/json"):
        if isinstance(payload, dict):
            payload = json.dumps(payload).encode()
        req = urllib.request.Request(base + path, data=payload,
                                     headers={"Content-Type": content_type} if payload is not None else {})
        try:
            response = urllib.request.urlopen(req, timeout=args.timeout)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, response.read()

    def upload(name, data):
        body, content_type = multipart({}, [("file", name + ".png", data)])
        status, raw = request("/api/upload", body, content_type)
        value = json.loads(raw)
        assert status == 200 and value.get("ok") and value.get("file"), value
        return value["file"]

    def log_since(offset):
        with server_log.open("rb") as log:
            log.seek(offset)
            text = log.read().decode("utf-8", "replace")
        return [line.strip() for line in text.splitlines()
                if "Qwen-Image-2.1:" in line or "keeping the source size" in line or "[qwen21-timing] total" in line]

    def finish(name, done, offset, started, expected, judge=None):
        assert not done.get("error"), done
        url = done["url"]
        status, data = request(url)
        assert status == 200, f"download failed: {status}"
        (output / f"{name}.png").write_bytes(data)
        width, height = png_size(data)
        case = {"name": name, "seconds": round(time.monotonic() - started, 1),
                "reported": [done.get("width"), done.get("height")], "png": [width, height],
                "sha256": hashlib.sha256(data).hexdigest(), "file": url.rsplit("/", 1)[-1],
                "server": log_since(offset)}
        if judge is not None:
            # A contrast case: what it shows is judged, not a size it must keep.
            case["passed"] = judge(width, height)
        else:
            case["passed"] = (width, height) == tuple(expected) and (done.get("width"), done.get("height")) == tuple(expected)
        case["expected"] = list(expected) if expected else None
        report["cases"].append(case)
        (output / "summary.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(case), flush=True)
        return case

    def stream(name, body, expected, judge=None):
        offset = server_log.stat().st_size
        started = time.monotonic()
        connection = http.client.HTTPConnection(args.host, args.port, timeout=args.timeout)
        connection.request("POST", "/api/image-edit/stream", json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 200, response.status
        done, raw = None, bytearray()
        try:
            for line in response:
                raw.extend(line)
                if line.startswith(b"data:"):
                    frame = json.loads(line[5:].strip())
                    if frame.get("done"):
                        done = frame
                        break
        finally:
            response.close()
            connection.close()
            (output / f"{name}.sse").write_bytes(raw)
        assert done, "no terminal frame"
        return finish(name, dict(done, request=body), offset, started, expected, judge)

    def multipart_edit(name, fields, image, expected):
        offset = server_log.stat().st_size
        started = time.monotonic()
        body, content_type = multipart({k: json.dumps(v) if not isinstance(v, str) else v for k, v in fields.items()},
                                       [("image", "source.png", image)])
        status, raw = request("/api/image-edit", body, content_type)
        value = json.loads(raw)
        assert status == 200, value
        return finish(name, value, offset, started, expected)

    server = None
    try:
        with server_log.open("w") as log:
            env = dict(os.environ, TENSORSHARP_UPLOAD_DIR=str(output / "server-uploads"))
            server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            report["server_pid"] = server.pid
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited during startup: {server.returncode}; see {server_log}")
                try:
                    with urllib.request.urlopen(base + "/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.5)
            else:
                raise TimeoutError("Server did not become healthy")

            size = tuple(args.size)
            source = upload("source", make_source(args.source, size, output / "source.png"))
            mask = upload("mask", make_mask(size, output / "mask.png"))
            common = {"steps": args.steps, "cfg": 1}

            # TensorAgent: a selection edit, then a change of its result, both at its 1 MP area.
            selected = stream("tensoragent-1-selection", dict(common, imagePaths=[source], maskPath=mask,
                              maskMode="grayscale", targetArea=TENSORAGENT_AREA, keepSourceSize=True, seed=first_seed,
                              prompt="Replace the snow in the selected area with green spring grass and small yellow flowers."),
                              size)
            change = dict(common, imagePaths=[selected["file"]], targetArea=TENSORAGENT_AREA, seed=second_seed,
                          prompt="Make the whole scene a warm golden sunset. Keep the fox unchanged.")
            stream("tensoragent-2-change", dict(change, keepSourceSize=True), size)
            if not args.skip_contrast:
                # The report's second picture: sized from the area alone, it comes back smaller.
                stream("tensoragent-2-change-without-keep", change, None,
                       judge=lambda width, height: width * height < size[0] * size[1])

            # Web UI: an edit, then an edit of its result (multipart), at the server's default area.
            first = stream("webui-1-edit", dict(common, imagePaths=[source], keepSourceSize=True, seed=first_seed,
                           prompt="Turn the background trees into a misty pine forest."), size)
            multipart_edit("webui-2-edit-multipart", dict(common, keepSourceSize=True, seed=second_seed,
                           prompt="Make it look like a watercolor painting."),
                           (output / f"{first['name']}.png").read_bytes(), size)

            if args.small_size:
                small_size = tuple(args.small_size)
                small = upload("small", make_source(args.source, small_size, output / "small.png"))
                body = dict(common, imagePaths=[small], targetArea=TENSORAGENT_AREA, seed=first_seed,
                            prompt="Make the whole scene a warm golden sunset. Keep the fox unchanged.")
                stream("small-keep", dict(body, keepSourceSize=True), small_size)
                # For comparison by eye: sampled at the full area, then brought to the same size.
                stream("small-without-keep", body, None, judge=lambda width, height: True)
                with Image.open(output / "small-without-keep.png") as image:
                    image.convert("RGB").resize(small_size, Image.LANCZOS).save(output / "small-without-keep-downscaled.png")
    finally:
        if server and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
        report["status"] = "passed" if report["cases"] and all(c.get("passed") for c in report["cases"]) else "failed"
        (output / "summary.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"status": report["status"], "output": str(output)}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
