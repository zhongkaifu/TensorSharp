"""Keep iOS Release link/strip roots aligned with TensorSharp native exports.

This checks the source manifest; it does not compile an Apple binary or certify
that symbols survived linking on an Apple device.
"""
import argparse
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Update the existing manifest; otherwise only check it.")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = root / "TensorSharp.GGML.Native"
    target = root / "TensorAgent/src/TensorAgent.Maui/GgmlExportedSymbols.targets"
    pattern = re.compile(r"^\s*TSG_EXPORT[^\r\n]*\b(TSGgml_[A-Za-z0-9_]+)\s*\(", re.MULTILINE)
    exports = sorted({
        name
        for file in source.iterdir()
        if file.suffix in (".cpp", ".c", ".inc", ".h")
        for name in pattern.findall(file.read_text(encoding="utf-8"))
        # Reserved diagnostic exports exist only with native test hooks enabled;
        # making them Release link roots would require absent archive symbols.
        if not name.startswith("TSGgml_Test")
    })
    document = ET.parse(target)
    retained = [element.attrib["Include"] for element in document.iter("ReferenceNativeSymbol")]
    missing = sorted(set(exports) - set(retained))
    extra = sorted(set(retained) - set(exports))
    duplicates = len(retained) != len(set(retained))
    if arguments.write:
        original = target.read_bytes()
        newline = "\r\n" if b"\r\n" in original else "\n"
        text = original.decode("utf-8")
        items = newline.join(f'    <ReferenceNativeSymbol Include="{name}" SymbolType="Function" />' for name in exports)
        text, groups = re.subn(
            r'(?P<open>  <ItemGroup Condition="[^"\r\n]*">)\r?\n.*?\r?\n(?P<close>  </ItemGroup>)',
            lambda match: match["open"] + newline + items + newline + match["close"],
            text, count=1, flags=re.DOTALL,
        )
        if groups != 1:
            raise RuntimeError("Expected exactly one native symbol ItemGroup.")
        text = re.sub(r"Native source export manifest; \d+ symbols\.", f"Native source export manifest; {len(exports)} symbols.", text)
        target.write_bytes(text.encode("utf-8"))
        print(f"Updated {len(exports)} native symbol roots ({len(missing)} added, {len(extra)} removed).")
        return 0
    if missing or extra or duplicates:
        print("iOS native symbol manifest is out of date.", file=sys.stderr)
        for name in missing:
            print("Missing: " + name, file=sys.stderr)
        for name in extra:
            print("Obsolete: " + name, file=sys.stderr)
        if duplicates:
            print("Duplicate symbol roots exist.", file=sys.stderr)
        return 1
    print(f"PASS: {len(exports)} native symbol roots match their source exports.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
