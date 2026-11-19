#!/usr/bin/env bash
# Task-level quality of one server: GSM8K (thinking off and on), the IFEval-style set and the
# function-calling set, all greedy, written as OUT_DIR/<set>-<LABEL>.jsonl (+ .log).
#
#   eng/lfm3/e2e/quality.sh URL LABEL OUT_DIR GSM8K_JSONL [N_NOTHINK] [N_THINK]
#
# Run it once per engine on the same GGUF and compare the pairs with compare.py, e.g.
#   compare.py OUT_DIR/gsm8k-nothink-ts.jsonl OUT_DIR/gsm8k-nothink-llama.jsonl
set -euo pipefail
url="$1"; label="$2"; out="$3"; data="$4"; n_off="${5:-200}"; n_on="${6:-50}"
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$out"
python3 -I "$here/gsm8k.py" --url "$url" --data "$data" --n "$n_off" \
  --out "$out/gsm8k-nothink-$label.jsonl" > "$out/gsm8k-nothink-$label.log" 2>&1
python3 -I "$here/gsm8k.py" --url "$url" --data "$data" --n "$n_on" --think --max-tokens 8192 \
  --out "$out/gsm8k-think-$label.jsonl" > "$out/gsm8k-think-$label.log" 2>&1
python3 -I "$here/ifeval.py" --url "$url" --out "$out/ifeval-$label.jsonl" > "$out/ifeval-$label.log" 2>&1
python3 -I "$here/toolcall_eval.py" --url "$url" --out "$out/tools-$label.jsonl" > "$out/tools-$label.log" 2>&1
python3 -I "$here/toolcall_eval.py" --url "$url" --stream --out "$out/tools-stream-$label.jsonl" > "$out/tools-stream-$label.log" 2>&1
for f in gsm8k-nothink gsm8k-think ifeval tools tools-stream; do
  echo "$f-$label: $(tail -1 "$out/$f-$label.log")"
done
