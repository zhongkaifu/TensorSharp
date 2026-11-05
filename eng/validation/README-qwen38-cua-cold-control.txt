One-request exact-body cold CUA diagnostic

qwen38-cua-cold-control.py is byte-for-byte the c48f996b177590e44a9d80dfe1ac150059268443dd341218bb9a90f8e78cdec2
helper used for the recorded cold controls. It reads a completed replay and its
ORIGINAL prepared fixture. It never creates a fixture, starts/stops a server,
discovers a model, warms a cache or alters frozen benchmark scoring.

CPU preparation only (no HTTP):

  python3 eng/validation/qwen38-cua-cold-control.py --harness-root . --replay docs/validation/RUN/replay.jsonl --fixtures artifacts/ORIGINAL_FIXTURE --step 5 --prepare-only --out artifacts/cold-step5-prepared.json

One actual request requires an already started fresh server and its launch
manifest. Root or the operator coordinates server ownership separately:

  python3 eng/validation/qwen38-cua-cold-control.py --harness-root . --replay docs/validation/RUN/replay.jsonl --manifest artifacts/FRESH_SERVER_MANIFEST.json --url http://127.0.0.1:5088 --step 5 --out artifacts/cold-step5.json

Defaults select scenario edit, lane0 and step5. Use --step 25 for the later
edit; --lanes disambiguates a source file when needed. --fixtures overrides
the source header's original path, but the original fixture SHA must match.
All generated outputs must remain in ignored artifacts/ or docs/validation/.
Existing output is never overwritten.

The helper restores the original frozen Client.body insertion order, verifies
EVERY request field and common/input/full-body hashes, and requires the exact
original wire-byte SHA. No field, model ID, message, image or tool is changed.
It also requires engine/mode/argv/working directory/context/lanes, complete
binary/model path-to-SHA inventories and selected mathematical policy envs to
match the original deployment. Fresh process identity must differ. Metadata
timestamps, source Git status and output metadata are not numerical identity.

The single POST uses frozen stream_chat/Client.finish and the original CUA
seven quality and three measurement checks. Qualification additionally needs
explicit actual cached_tokens=0 and unchanged positive prompt usage. Missing,
boolean or nonzero cache counts never pass. The complete response, consumed
SSE bytes, original request bytes, usage, hashes and quality FAIL are retained.
Tool-parser SSE is not a target-token stream; no decode token/s is reported.

complete means transport/execution completeness, not quality PASS. Exit1 can
be an entirely recorded cold response that still fails the original quality
check. Read diagnostic_qualified, cold_gate and quality separately.

This is one exact historical request, NOT a new30-step benchmark or an
independent replay run. Warm/cold execution can select different prefill
grouping and numerical paths; output differences alone do not prove cache
corruption. A cold failure also does not prove a particular cache mechanism.
It does not regrade the original scenario, establish broad visual quality or
authorize a performance ranking. Keep the original failures and evidence.

Standalone CPU tests generate small fixtures and PNGs under TemporaryDirectory,
build mock completed source rows and perform mock POSTs only. They do not read
ignored actual evidence/fixtures and have no model/device/network dependency:

  python3 -m unittest discover -s eng/validation/tests -p test_qwen38_cua_cold_control.py -v
