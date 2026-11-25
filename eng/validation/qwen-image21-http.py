#!/usr/bin/env python3
"""Exercise a real Qwen-Image-2.1 Server.Host process over HTTP.

These 128px, 1–2 step requests verify transport, state reuse, previews, alpha,
and error handling. They are not image-quality or throughput benchmarks.
The script starts one server, runs requests serially, saves evidence in the
ignored validation directory, and terminates only the process it started.

  python3 eng/validation/qwen-image21-http.py --dry-run
  python3 eng/validation/qwen-image21-http.py --models ../models
"""
import argparse
import base64
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
import urllib.parse
import urllib.request
import uuid
import zlib

ROOT = Path(__file__).resolve().parents[2]


def png_metadata(data):
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise AssertionError("Response is not a PNG with an IHDR header")
    width, height, depth, color, compression, filter_method, interlace = struct.unpack(
        ">IIBBBBB", data[16:29])
    # Validate all CRCs and require a complete stream, not merely its header.
    offset, image_data, ended = 8, bytearray(), False
    while offset < len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        crc = struct.unpack(">I", data[offset + 8 + length:offset + 12 + length])[0]
        assert zlib.crc32(kind + payload) & 0xffffffff == crc, "PNG CRC mismatch"
        if kind == b"IDAT":
            image_data.extend(payload)
        if kind == b"IEND":
            ended = True
            break
        offset += length + 12
    assert ended and image_data, "Incomplete PNG"
    inflated = zlib.decompress(image_data)
    assert inflated, "PNG has no image samples"
    return {"width": width, "height": height, "bit_depth": depth, "color_type": color,
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def fixture_png(size, alternate=False):
    """A deterministic RGBA protocol fixture, including fully transparent pixels."""
    def chunk(kind, content):
        return (struct.pack(">I", len(content)) + kind + content +
                struct.pack(">I", zlib.crc32(kind + content) & 0xffffffff))
    rows = bytearray()
    for y in range(size):
        rows.append(0)
        for x in range(size):
            inside = size // 4 <= x < size * 3 // 4 and size // 4 <= y < size * 3 // 4
            color = (35, 85, 220) if alternate else (220, 45, 35)
            rows.extend((*color, 255 if inside else 0))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(bytes(rows))) + chunk(b"IEND", b""))


def multipart(fields, files):
    boundary = "qwen21-" + uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
                      f'{value}\r\n').encode())
    for name, filename, content in files:
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
                      f'filename="{filename}"\r\nContent-Type: image/png\r\n\r\n').encode() + content + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), "multipart/form-data; boundary=" + boundary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT.parent / "models")
    parser.add_argument("--dit", type=Path, help="Use this GGUF instead of the download configuration.")
    parser.add_argument("--vae", type=Path)
    parser.add_argument("--text-encoder", type=Path)
    parser.add_argument("--mmproj", type=Path)
    parser.add_argument("--backend", default="ggml_metal",
                        choices=("ggml_metal", "ggml_cuda", "ggml_cpu", "ggml_vulkan"))
    parser.add_argument("--output", type=Path, default=ROOT / "docs/validation/qwen-image-2.1/http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5017)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.size < 32 or args.size % 32:
        parser.error("--size must be a positive multiple of 32")
    executable = ROOT / "TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll"
    command = ["dotnet", str(executable), "--config", "config/qwen-image-2.1.json",
               "--host", args.host, "--port", str(args.port)]
    if args.dit:
        companions = (("--qwen-image-vae", args.vae), ("--qwen-image-vl", args.text_encoder),
                      ("--qwen-image-mmproj", args.mmproj))
        if any(path is None for _, path in companions):
            parser.error("--dit requires --vae, --text-encoder and --mmproj to test all editing cases.")
        command = ["dotnet", str(executable), "--model", str(args.dit.resolve()),
                   "--host", args.host, "--port", str(args.port)]
        for flag, path in companions:
            command += [flag, str(path.resolve())]
        if not args.dry_run:
            for path in (args.dit, args.vae, args.text_encoder, args.mmproj):
                if not path.is_file():
                    parser.error(f"Required model file is missing: {path}")
    elif any(path is not None for path in (args.vae, args.text_encoder, args.mmproj)):
        parser.error("Explicit companions require --dit.")
    command += ["--backend", args.backend]
    cases = ["health", "models", "invalid_dimensions", "invalid_prompt", "stream_error",
             "upload_rgba", "upload_second_reference", "generate_json", "edit_json", "edit_multipart",
             "edit_two_references", "generate_stream_preview", "edit_stream_preview",
             "generate_repeated", "generate_eight_step_previews", "cancel_stream", "generate_after_cancel"]
    if args.dry_run:
        print(json.dumps({"command": command, "environment": {"TENSORSHARP_MODELS": str(args.models.resolve()),
                          "TENSORSHARP_UPLOAD_DIR": str(args.output.resolve() / "server-uploads")},
                          "output": str(args.output.resolve()), "cases": cases}, indent=2))
        return 0
    if not executable.is_file():
        parser.error("Build TensorSharp.Server.Host before running this harness")
    with socket.socket() as probe:
        if probe.connect_ex((args.host, args.port)) == 0:
            parser.error("Port is already in use; refusing to test or terminate an unrelated server")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = f"http://{args.host}:{args.port}"
    report = {"purpose": "HTTP execution coverage, not quality/performance validation", "command": command,
              "models": str(args.models.resolve()), "size": args.size, "cases": [],
              "server_dll_sha256": hashlib.sha256(executable.read_bytes()).hexdigest()}
    server = None
    server_log = output / "server.log"

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
            return response.status, response.read(), dict(response.headers)

    def json_request(name, path, payload=None, status=200, content_type="application/json"):
        code, raw, headers = request(path, payload, content_type)
        (output / (name + ".response.json")).write_bytes(raw)
        assert code == status, f"Expected HTTP {status}, got {code}: {raw[:1000]!r}"
        value = json.loads(raw)
        if status >= 400:
            assert value.get("error"), "Refusal must include useful error feedback"
        return value

    def image_result(name, value):
        assert value.get("url"), f"No final download URL: {value}"
        assert (value.get("width"), value.get("height")) == (args.size, args.size)
        url = urllib.parse.urlsplit(value["url"])
        assert not url.netloc or url.netloc == f"{args.host}:{args.port}", "Unexpected external result URL"
        status, data, _ = request(url.path + ("?" + url.query if url.query else ""))
        assert status == 200, f"Image download failed: HTTP {status}"
        info = png_metadata(data)
        assert (info["width"], info["height"]) == (args.size, args.size)
        assert info["color_type"] == 6, "Qwen-Image-2.1 output must retain RGBA"
        (output / (name + ".png")).write_bytes(data)
        return {"image": info, "elapsed_seconds_reported": value.get("elapsedSeconds")}

    def stream(name, path, body, error=False, cancel=False, expected_previews=None):
        started = time.monotonic()
        connection = http.client.HTTPConnection(args.host, args.port, timeout=args.timeout)
        connection.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 200
        assert "text/event-stream" in response.getheader("Content-Type", "")
        frames, raw, first_time = [], bytearray(), None
        try:
            while True:
                line = response.readline()
                if not line:
                    break
                raw.extend(line)
                if not line.startswith(b"data:"):
                    continue
                frame = json.loads(line[5:].strip())
                frames.append(frame)
                if first_time is None:
                    first_time = time.monotonic() - started
                if cancel and frame.get("step"):
                    # Abort the transport while inference is active, then verify
                    # that the same server successfully handles the next request.
                    if connection.sock:
                        connection.sock.shutdown(socket.SHUT_RDWR)
                    break
                if frame.get("done"):
                    break
        finally:
            response.close()
            connection.close()
            (output / (name + ".sse")).write_bytes(raw)
        if cancel:
            assert frames and frames[-1].get("step") == 1 and not frames[-1].get("done")
            return {"disconnected_after_step": 1, "requested_steps": body["steps"],
                    "first_frame_seconds": first_time, "recovery_case": "generate_after_cancel"}
        assert frames and frames[-1].get("done"), "Missing terminal SSE frame"
        if error:
            assert frames[-1].get("error"), "SSE error must be delivered in a terminal frame"
            return {"frames": len(frames), "error": frames[-1]["error"]}
        assert not frames[-1].get("error"), frames[-1]
        progress = [frame for frame in frames if frame.get("step")]
        assert [frame["step"] for frame in progress] == list(range(1, body["steps"] + 1))
        assert all(frame.get("total") == body["steps"] for frame in progress)
        previews = [frame for frame in progress if frame.get("image")]
        assert previews, "Streaming request must include an intermediate preview"
        if expected_previews is not None:
            assert len(previews) == expected_previews, f"Expected {expected_previews} finite previews, got {len(previews)}"
        for i, frame in enumerate(previews):
            prefix, encoded = frame["image"].split(",", 1)
            assert prefix == "data:image/png;base64"
            data = base64.b64decode(encoded, validate=True)
            metadata = png_metadata(data)
            assert metadata["width"] == frame["width"] and metadata["height"] == frame["height"]
            assert metadata["color_type"] == 6
            (output / f"{name}.preview-{i + 1}.png").write_bytes(data)
        return {"frames": len(frames), "preview_count": len(previews), "first_frame_seconds": first_time,
                **image_result(name, frames[-1])}

    def run(name, function):
        began = time.monotonic()
        try:
            details = function()
            case = {"name": name, "passed": True, "seconds": time.monotonic() - began, "details": details}
        except Exception as error:
            case = {"name": name, "passed": False, "seconds": time.monotonic() - began,
                    "error": f"{type(error).__name__}: {error}"}
        report["cases"].append(case)
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(case), flush=True)
        return case.get("details") if case["passed"] else None

    options = {"prompt": "A red ceramic cube on a white table.", "width": args.size, "height": args.size,
               "steps": 2, "cfg": 1, "seed": 42, "negativePrompt": " "}
    first_png = fixture_png(args.size)
    second_png = fixture_png(args.size, alternate=True)
    (output / "reference-rgba.png").write_bytes(first_png)
    (output / "reference-second.png").write_bytes(second_png)
    try:
        env = dict(os.environ, TENSORSHARP_MODELS=str(args.models.resolve()),
                   TENSORSHARP_UPLOAD_DIR=str(output / "server-uploads"))
        with server_log.open("w") as log:
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

            def health():
                status = request("/health")[0]
                assert status == 200, f"Health endpoint returned HTTP {status}"
                return {"status": status}
            def models():
                value = json_request("models", "/api/models")
                assert value.get("architecture") == "qwen_image", value
                assert value.get("loadedBackend") == args.backend, value
                if args.dit:
                    assert value.get("loaded") == args.dit.name, value
                return value
            run("health", health)
            run("models", models)
            run("invalid_dimensions", lambda: json_request("invalid_dimensions", "/api/image-generate",
                dict(options, width=args.size + 1), status=400))
            run("invalid_prompt", lambda: json_request("invalid_prompt", "/api/image-generate",
                dict(options, prompt=""), status=400))
            run("stream_error", lambda: stream("stream_error", "/api/image-generate/stream",
                dict(options, width=args.size + 1), error=True))

            def upload(name, data):
                body, content_type = multipart({}, [("file", name + ".png", data)])
                value = json_request(name, "/api/upload", body, content_type=content_type)
                assert value.get("ok") and value.get("file") and value.get("mediaType") == "image"
                return value
            reference = run("upload_rgba", lambda: upload("upload_rgba", first_png))
            second = run("upload_second_reference", lambda: upload("upload_second_reference", second_png))
            initial = run("generate_json", lambda: image_result("generate_json",
                json_request("generate_json", "/api/image-generate", options)))
            if reference:
                edit = dict(options, prompt="Make the red square green.", steps=1, imagePaths=[reference["file"]])
                run("edit_json", lambda: image_result("edit_json", json_request("edit_json", "/api/image-edit", edit)))
                def multipart_edit():
                    body, content_type = multipart(dict(options, steps=1, prompt="Make the square green."),
                        [("image", "reference.png", first_png)])
                    return image_result("edit_multipart", json_request("edit_multipart", "/api/image-edit",
                        body, content_type=content_type))
                run("edit_multipart", multipart_edit)
                if second:
                    run("edit_two_references", lambda: image_result("edit_two_references",
                        json_request("edit_two_references", "/api/image-edit",
                            dict(edit, imagePaths=[reference["file"], second["file"]],
                                 prompt="Place the squares from image 1 and image 2 side by side."))))
                run("edit_stream_preview", lambda: stream("edit_stream_preview", "/api/image-edit/stream", dict(edit, steps=2)))
            run("generate_stream_preview", lambda: stream("generate_stream_preview", "/api/image-generate/stream", options))
            def repeated():
                value = image_result("generate_repeated", json_request("generate_repeated", "/api/image-generate", options))
                assert initial and value["image"]["sha256"] == initial["image"]["sha256"], "Same-process repeated seeded result changed"
                return value
            run("generate_repeated", repeated)
            run("generate_eight_step_previews", lambda: stream("generate_eight_step_previews",
                "/api/image-generate/stream", dict(options, prompt="A blue cube.", steps=8), expected_previews=7))
            run("cancel_stream", lambda: stream("cancel_stream", "/api/image-generate/stream",
                dict(options, prompt="Cancel HTTP probe: a blue cube.", steps=8), cancel=True))
            run("generate_after_cancel", lambda: image_result("generate_after_cancel",
                json_request("generate_after_cancel", "/api/image-generate", dict(options, steps=1))))
            report["server_alive_after_requests"] = server.poll() is None
    except Exception as error:
        report["fatal_error"] = f"{type(error).__name__}: {error}"
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=10)
        if server_log.exists():
            text = server_log.read_text(errors="replace")
            marker = text.find("Cancel HTTP probe:")
            if marker >= 0:
                cancel_section = text[marker:]
                # Only the cancelled request has eight steps; the recovery has
                # one. Its HTTP log may appear while waiting for the model lock.
                # The diffusion runtime logs "[diffusion] qwen_image step k/8 ..."; older builds "[qwen21-step] k/8".
                steps = [int(x) for x in re.findall(r"(?:\[qwen21-step\] |\[diffusion\] qwen_image step )(\d+)/8", cancel_section)]
                report["cancellation_logged_steps"] = steps
                report["cancellation_stopped_before_final_step"] = bool(steps) and max(steps) < 8
        completed_names = {case["name"] for case in report["cases"]}
        report["unexecuted_cases"] = [name for name in cases if name not in completed_names]
        report["passed"] = (not report.get("fatal_error") and not report["unexecuted_cases"] and
                            all(case["passed"] for case in report["cases"]) and
                            report.get("server_alive_after_requests", False) and
                            report.get("cancellation_stopped_before_final_step", False))
        (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"passed": report["passed"], "report": str(output / "report.json")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
