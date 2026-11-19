#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
#
# This file is part of TensorSharp.
#
# TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
"""Spoken web chat replies against a running TensorSharp.Server.Host with LFM3-Omni and its mmproj loaded.

Checks /api/models' speechOutput, then a session of spoken /api/chat turns: the frame order (token and audio frames
interleaved, a done frame with audioId / audioUrl / audioSeconds), that the WAV the done frame names downloads, parses
and is as long as the audio frames, and what each later turn reused (kvReusedTokens) when the history carries the
spoken turns back by their audioId. A last text turn shows the reuse when the next turn is not spoken. Writes a JSON
report (and the WAVs) to --output, which must be under the ignored artifacts/ or docs/validation/.

  python3 eng/lfm3/speech-webchat-check.py --url http://127.0.0.1:18765 --output artifacts/lfm3-speech-webchat [--lang zh]
"""
import argparse
import base64
import json
import os
import struct
import subprocess
import sys
import urllib.request

QUESTIONS = {
    "en": [
        "What is the capital of France? Answer in one short sentence.",
        "And what is the capital of Germany? Answer in one short sentence.",
        "Which of the two cities did you name first?",
    ],
    # Chinese: LFM3-Omni speaks these replies from their text (its interleaved mode speaks English only).
    "zh": [
        "法国的首都是哪里？请用一句话回答。",
        "那德国的首都呢？请用一句话回答。",
        "你先说的是哪个城市？",
    ],
}


def request(url, body=None, method=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method or ("POST" if body is not None else "GET"),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)


def chat(base, body):
    frames = []
    with request(base + "/api/chat", body) as resp:
        for raw in resp:
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith("data: "):
                frames.append(json.loads(line[6:]))
    return frames


def wav_info(data):
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    channels, rate = struct.unpack_from("<HI", data, 22)
    bits = struct.unpack_from("<H", data, 34)[0]
    size = struct.unpack_from("<I", data, 40)[0]
    return {"channels": channels, "rate": rate, "bits": bits, "dataBytes": size,
            "seconds": size / (rate * channels * bits / 8)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--voice", default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--lang", default="en", choices=sorted(QUESTIONS), help="the questions' language")
    args = parser.parse_args()
    out = os.path.abspath(args.output)
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if not any(out.startswith(os.path.join(repo, d) + os.sep) for d in ("artifacts", "docs/validation")):
        sys.exit("--output must be under the ignored artifacts/ or docs/validation/ folders")
    os.makedirs(out, exist_ok=True)
    base = args.url.rstrip("/")
    report = {"url": base, "turns": []}
    failures = []

    models = json.load(request(base + "/api/models"))
    report["speechOutput"] = models.get("speechOutput")
    report["model"] = models.get("loaded")
    report["mmproj"] = models.get("loadedMmProj")
    report["backend"] = models.get("loadedBackend")
    if not (models.get("speechOutput") or {}).get("available"):
        failures.append("speechOutput is not available: %r" % models.get("speechOutput"))

    session = json.load(request(base + "/api/sessions", {}))["sessionId"]
    history = []
    speech = {"voice": args.voice} if args.voice else True
    questions = QUESTIONS[args.lang]
    for index, question in enumerate(questions + ["Thanks! (text turn)"]):
        spoken = index < len(questions)
        history.append({"role": "user", "content": question})
        body = {"sessionId": session, "messages": history, "maxTokens": args.max_tokens}
        if spoken:
            body["speech"] = speech
        frames = chat(base, body)
        done = frames[-1] if frames and frames[-1].get("done") else None
        order = "".join("t" if "token" in f else "a" if "audio" in f else "n" if "notice" in f else "d" if f.get("done") else "?"
                        for f in frames)
        text = "".join(f.get("token", "") for f in frames)
        pcm = b"".join(base64.b64decode(f["audio"]) for f in frames if "audio" in f)
        rates = sorted({f.get("sampleRate") for f in frames if "audio" in f})
        turn = {"spoken": spoken, "question": question, "text": text, "frames": len(frames),
                "audioFrames": order.count("a"), "tokenFrames": order.count("t"),
                "order": order if len(order) <= 400 else order[:400] + "...", "sampleRates": rates,
                "streamedAudioSeconds": len(pcm) / 2 / rates[0] if rates else 0}
        if done is None:
            failures.append("turn %d ended without a done frame" % index)
            report["turns"].append(turn)
            break
        for key in ("promptTokens", "kvReusedTokens", "kvReusePercent", "tokenCount", "elapsed", "error",
                    "audioId", "audioUrl", "audioSeconds"):
            turn[key] = done.get(key)
        if done.get("error"):
            failures.append("turn %d failed: %s" % (index, done["error"]))
        if spoken:
            if "a" not in order or "t" not in order:
                failures.append("turn %d: no interleaved token/audio frames (%s)" % (index, order[:60]))
            if not done.get("audioId") or not done.get("audioUrl"):
                failures.append("turn %d: done frame has no audioId/audioUrl" % index)
            else:
                data = request(base + done["audioUrl"]).read()
                name = os.path.join(out, "turn%d.wav" % (index + 1))
                with open(name, "wb") as f:
                    f.write(data)
                info = wav_info(data)
                turn["wav"] = info
                turn["wavFile"] = name
                if info["dataBytes"] != len(pcm):
                    failures.append("turn %d: WAV holds %d bytes of PCM, the audio frames %d" % (index, info["dataBytes"], len(pcm)))
                if info["seconds"] <= 0:
                    failures.append("turn %d: WAV is empty" % index)
                try:
                    afinfo = subprocess.run(["afinfo", name], capture_output=True, text=True, timeout=30)
                    turn["afinfo"] = [l.strip() for l in afinfo.stdout.splitlines()
                                      if "estimated duration" in l or "Data format" in l]
                except (OSError, subprocess.TimeoutExpired) as e:
                    turn["afinfo"] = "unavailable: %s" % e
        message = {"role": "assistant", "content": text}
        if done.get("audioId"):
            message["audioId"] = done["audioId"]
        if done.get("audioUrl"):
            message["audioUrl"] = done["audioUrl"]
            message["audioSeconds"] = done.get("audioSeconds")
        history.append(message)
        report["turns"].append(turn)
        print("turn %d %s: prompt=%s reused=%s audio=%s s text=%r" % (
            index + 1, "spoken" if spoken else "text", turn.get("promptTokens"), turn.get("kvReusedTokens"),
            turn.get("audioSeconds"), text[:80]), flush=True)

    for index, turn in enumerate(report["turns"][1:len(questions)], start=1):
        if not turn.get("kvReusedTokens"):
            failures.append("spoken turn %d reused no cached prompt tokens" % (index + 1))

    report["failures"] = failures
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"failures": failures}, indent=2))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
