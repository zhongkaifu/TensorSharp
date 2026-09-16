#!/usr/bin/env python3
"""Exercise embedding context admission, truncation and recovery over real HTTP."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/EmbeddingBench"))
import embedding_bench


def run(url, model, context_limit, output, timeout):
    report = {"model": model, "context_limit": context_limit, "cases": [],
              "status": "running", "started_at_unix": time.time(),
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Context boundary mechanics with repeated text; not long-document retrieval quality."}
    session = requests.Session()

    def save():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")

    def request(name, path, body, expected_status, expected_tokens=None, reference=None):
        encoded = json.dumps(body, ensure_ascii=False, sort_keys=True).encode()
        case = {"name": name, "path": path, "input_sha256": hashlib.sha256(encoded).hexdigest(),
                "request": body, "expected_status": expected_status, "status": "failed"}
        started = time.perf_counter()
        try:
            response = session.post(url.rstrip("/") + path, json=body, timeout=timeout)
            case.update(http_status=response.status_code, response=response.json())
            assert response.status_code == expected_status, (response.status_code, case["response"])
            if expected_status == 200:
                data = case["response"]
                vectors = ([row["embedding"] for row in data["data"]]
                           if path == "/v1/embeddings" else data["embeddings"])
                tokens = (data["usage"]["prompt_tokens"] if path == "/v1/embeddings"
                          else data["prompt_eval_count"])
                assert len(vectors) == 1 and vectors[0], "Expected one nonempty embedding"
                embedding_bench.assert_vectors(vectors, len(vectors[0]))
                assert tokens == expected_tokens, (tokens, expected_tokens)
                case.update(prompt_tokens=tokens, dimensions=len(vectors[0]))
                if reference is not None:
                    cosine = embedding_bench.cosine(vectors[0], reference)
                    error = max(abs(a - b) for a, b in zip(vectors[0], reference))
                    case.update(reference_cosine=cosine, max_component_error=error)
                    assert cosine >= .99999 and error <= 1e-5, (cosine, error)
                case["status"] = "passed"
                return vectors[0]
            assert case["response"].get("error"), "Error response has no error payload"
            case["status"] = "passed"
        except Exception as error:
            case["error"] = str(error)
        finally:
            case["wall_seconds"] = time.perf_counter() - started
            report["cases"].append(case)
            save()
        return None

    try:
        # Both catalog encoders use BERT-family tokenizers. Verify the fixture's
        # one-token word and two special tokens rather than assuming its length.
        probe = request("token-count-probe", "/v1/embeddings", {"model": model, "input": "cat cat cat"}, 200, 5)
        if probe is None:
            raise ValueError("Fixture token-count probe failed; boundary inputs are unqualified")
        exact = " ".join(["cat"] * (context_limit - 2))
        overflow = exact + " cat"
        boundary = request("exact-context-limit", "/v1/embeddings", {"model": model, "input": exact}, 200, context_limit)
        request("openai-one-token-over", "/v1/embeddings", {"model": model, "input": overflow}, 400)
        request("ollama-no-truncation-over", "/api/embed", {"model": model, "input": overflow, "truncate": False}, 400)
        if boundary is None:
            raise ValueError("Exact-limit embedding failed; truncation comparison is unqualified")
        request("ollama-default-truncation", "/api/embed", {"model": model, "input": overflow}, 200, context_limit, boundary)
        request("ollama-explicit-truncation", "/api/embed", {"model": model, "input": overflow, "truncate": True}, 200, context_limit, boundary)
        request("recovery-after-overflow", "/v1/embeddings", {"model": model, "input": "cat cat cat"}, 200, 5, probe)
    except Exception as error:
        report["error"] = str(error)
    finally:
        session.close()
        report["finished_at_unix"] = time.time()
        report["status"] = ("passed" if not report.get("error") and len(report["cases"]) == 7
                            and all(case["status"] == "passed" for case in report["cases"]) else "failed")
        save()
    print(report["status"], [(case["name"], case["status"]) for case in report["cases"]], flush=True)
    return int(report["status"] != "passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--context-limit", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.context_limit < 5 or args.timeout <= 0:
        parser.error("context-limit must be at least 5 and timeout positive")
    return run(args.url, args.model, args.context_limit, args.output, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
