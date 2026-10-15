"""Verify a validation checkout against a SHA-256 source manifest.

The manifest is a JSON array of {"Path": "repo/relative/file", "Sha256": "..."}.
This verifies source identity only, not that a binary was rebuilt from it.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    mismatches = []
    for item in manifest:
        relative = Path(item["Path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError(f"Manifest path leaves the checkout: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != item["Sha256"].lower():
            mismatches.append({**item, "Actual": actual})
    result = {"ComparedFiles": len(manifest), "Mismatches": mismatches}
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return int(not manifest or bool(mismatches))


if __name__ == "__main__":
    raise SystemExit(main())
