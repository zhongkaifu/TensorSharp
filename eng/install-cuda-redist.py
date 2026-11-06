#!/usr/bin/env python3
"""Assemble a CUDA toolkit from NVIDIA's per-component redistributable archives.

NVIDIA publishes every CUDA release as component archives (nvcc, crt, cudart,
cuBLAS, ...) listed with their SHA-256 in
https://developer.download.nvidia.com/compute/cuda/redist/redistrib_<version>.json.
Installing only those components gives a build machine everything the TensorSharp
native builds need, without the driver packages the full installers pull in, and
works for releases the setup-cuda actions do not list yet (the 13.4 Windows
installers are not published under the usual URLs at all).

Each archive holds one top-level directory with bin/, include/, lib/ (or nvvm/)
below it; those trees are merged into --dest, which then looks like an ordinary
toolkit root (CUDA_PATH). Only the Python standard library is used.

    python eng/install-cuda-redist.py --version 13.4.3 --dest "$RUNNER_TEMP/cuda" --github-env
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile


DEFAULT_BASE_URL = "https://developer.download.nvidia.com/compute/cuda/redist"
# What TensorSharp.GGML.Native (ggml-cuda: CCCL, cudart, cuBLAS, the driver stub
# shipped in cuda_cudart) and the TensorSharp.Backends.Cuda kernel compile need.
DEFAULT_COMPONENTS = ("cuda_nvcc", "cuda_crt", "libnvvm", "cuda_cudart", "libcublas", "cccl")
# Components NVIDIA renamed between releases: the manifests of CUDA 13.0.x and 13.2.x
# call CCCL "cuda_cccl", 13.4.x calls it "cccl". Either spelling installs whichever
# one the requested release lists, so the default list works for every CUDA 13.
COMPONENT_ALIASES = {
    "cccl": ("cccl", "cuda_cccl"),
    "cuda_cccl": ("cuda_cccl", "cccl"),
}


def detect_platform() -> str:
    machine = platform.machine().lower()
    if sys.platform.startswith("win"):
        # platform.machine() is the interpreter's view: an x64 Python under
        # emulation on Windows on Arm reports AMD64 and gets the x64 toolkit.
        return "windows-arm64" if machine in ("arm64", "aarch64") else "windows-x86_64"
    if sys.platform.startswith("linux"):
        return "linux-sbsa" if machine in ("aarch64", "arm64") else "linux-x86_64"
    raise SystemExit(f"Unsupported host platform {sys.platform}/{machine}; pass --platform.")


def fetch(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url, timeout=600) as response, open(destination, "wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_components(manifest: dict, names: list[str], target: str) -> list[tuple[str, str, str]]:
    """(name, relative_path, sha256) for each requested component; every one must exist.

    A renamed component (COMPONENT_ALIASES) resolves to the spelling this manifest
    uses, and the returned name is that spelling.
    """
    resolved = []
    for requested in names:
        name = next((alias for alias in COMPONENT_ALIASES.get(requested, (requested,))
                     if isinstance(manifest.get(alias), dict)), requested)
        entry = manifest.get(name)
        if not isinstance(entry, dict):
            raise SystemExit(f"Component '{requested}' is not part of CUDA {manifest.get('release_label', '?')}.")
        build = entry.get(target)
        if not isinstance(build, dict) or "relative_path" not in build or "sha256" not in build:
            raise SystemExit(f"Component '{name}' has no {target} archive.")
        resolved.append((name, build["relative_path"], build["sha256"]))
    return resolved


def _stripped(member_name: str) -> str | None:
    """The member's path below the archive's single top-level directory."""
    parts = member_name.replace("\\", "/").split("/", 1)
    if len(parts) < 2 or not parts[1]:
        return None
    relative = parts[1]
    if relative.startswith("/") or ".." in relative.split("/"):
        raise SystemExit(f"Refusing archive member outside the toolkit root: {member_name}")
    return relative


def extract_into(archive: Path, dest: Path) -> int:
    count = 0
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                relative = _stripped(info.filename)
                if relative is None or info.is_dir():
                    continue
                target = dest / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as source, open(target, "wb") as out:
                    shutil.copyfileobj(source, out)
                mode = (info.external_attr >> 16) & 0o777
                if mode:
                    os.chmod(target, mode)
                count += 1
        return count
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            relative = _stripped(member.name)
            if relative is None or member.isdir():
                continue
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.issym():
                if os.path.isabs(member.linkname) or ".." in Path(member.linkname).parts:
                    raise SystemExit(f"Refusing symlink outside its directory: {member.name} -> {member.linkname}")
                if target.is_symlink() or target.exists():
                    target.unlink()
                os.symlink(member.linkname, target)
            elif member.isfile():
                source = bundle.extractfile(member)
                with source, open(target, "wb") as out:
                    shutil.copyfileobj(source, out)
                os.chmod(target, member.mode & 0o777)
            else:
                continue
            count += 1
    return count


def install(version: str, dest: Path, components: list[str], target: str, base_url: str, cache: Path | None) -> None:
    base_url = base_url.rstrip("/")
    work = Path(tempfile.mkdtemp(prefix="cuda-redist-")) if cache is None else cache
    work.mkdir(parents=True, exist_ok=True)
    manifest_path = work / f"redistrib_{version}.json"
    fetch(f"{base_url}/redistrib_{version}.json", manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dest.mkdir(parents=True, exist_ok=True)
    resolved = resolve_components(manifest, components, target)
    for name, relative_path, expected in resolved:
        archive = work / Path(relative_path).name
        if not archive.exists() or sha256_of(archive) != expected:
            print(f"Downloading {name}: {relative_path}", flush=True)
            fetch(f"{base_url}/{relative_path}", archive)
        actual = sha256_of(archive)
        if actual != expected:
            raise SystemExit(f"SHA-256 mismatch for {relative_path}: expected {expected}, got {actual}.")
        files = extract_into(archive, dest)
        print(f"Installed {name} ({files} files)", flush=True)
    # Linux toolkits keep their libraries in lib64 (a lib64 -> targets/... link in
    # the packaged installs); the redist archives use lib/. Tools that only probe
    # lib64 then still find them.
    if target.startswith("linux") and (dest / "lib").is_dir() and not (dest / "lib64").exists():
        os.symlink("lib", dest / "lib64")
    (dest / "tensorsharp-cuda-redist.json").write_text(
        json.dumps({"version": version, "platform": target, "components": [name for name, _, _ in resolved]},
                   indent=2) + "\n",
        encoding="utf-8")


def export_github_env(dest: Path, target: str) -> None:
    env_file = os.environ.get("GITHUB_ENV")
    path_file = os.environ.get("GITHUB_PATH")
    if not env_file or not path_file:
        raise SystemExit("--github-env needs GITHUB_ENV and GITHUB_PATH (run it inside a GitHub Actions step).")
    root = str(dest.resolve())
    with open(env_file, "a", encoding="utf-8") as env:
        env.write(f"CUDA_PATH={root}\nCUDA_HOME={root}\n")
    with open(path_file, "a", encoding="utf-8") as paths:
        paths.write(str(dest.resolve() / "bin") + "\n")
        # CUDA 13 on Windows keeps its DLLs in a per-architecture subdirectory of
        # bin (bin\x64); add whichever the installed components created.
        if target.startswith("windows"):
            for arch_dir in ("x64", "arm64"):
                if (dest / "bin" / arch_dir).is_dir():
                    paths.write(str(dest.resolve() / "bin" / arch_dir) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help="CUDA release, e.g. 13.4.3")
    parser.add_argument("--dest", required=True, type=Path, help="toolkit root to create (becomes CUDA_PATH)")
    parser.add_argument("--components", default=",".join(DEFAULT_COMPONENTS),
                        help="comma-separated redist component names (default: %(default)s)")
    parser.add_argument("--platform", default=None,
                        help="redist platform key: linux-x86_64, linux-sbsa, windows-x86_64 or windows-arm64 "
                             "(default: this host)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--cache", type=Path, default=None, help="keep downloaded archives here for reuse")
    parser.add_argument("--github-env", action="store_true",
                        help="export CUDA_PATH/CUDA_HOME and add bin (and bin\\x64 or bin\\arm64 on Windows) to PATH "
                             "for later steps")
    args = parser.parse_args()
    target = args.platform or detect_platform()
    components = [c.strip() for c in args.components.split(",") if c.strip()]
    install(args.version, args.dest, components, target, args.base_url, args.cache)
    if args.github_env:
        export_github_env(args.dest, target)
    print(f"CUDA {args.version} ({target}) installed at {args.dest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
