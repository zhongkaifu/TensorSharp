"""Checks eng/install-cuda-redist.py against local fixture archives; nothing is downloaded."""

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile


ENG = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("install_cuda_redist", ENG / "install-cuda-redist.py")
redist = importlib.util.module_from_spec(spec)
spec.loader.exec_module(redist)


class InstallCudaRedistTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cuda-redist-fixture-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.server = self.root / "redist"
        self.server.mkdir()
        self.manifest = {"release_label": "99.1.0"}

    def add_zip(self, name, files):
        path = self.server / name / "windows-x86_64" / f"{name}-windows-x86_64-1.0-archive.zip"
        path.parent.mkdir(parents=True)
        with zipfile.ZipFile(path, "w") as bundle:
            for relative, data in files.items():
                bundle.writestr(f"{name}-windows-x86_64-1.0-archive/{relative}", data)
        self.register(name, "windows-x86_64", path)

    def add_tar(self, name, files, links=None):
        path = self.server / name / "linux-x86_64" / f"{name}-linux-x86_64-1.0-archive.tar.xz"
        path.parent.mkdir(parents=True)
        top = f"{name}-linux-x86_64-1.0-archive"
        with tarfile.open(path, "w:xz") as bundle:
            for relative, data in files.items():
                info = tarfile.TarInfo(f"{top}/{relative}")
                info.size = len(data)
                info.mode = 0o755
                bundle.addfile(info, io.BytesIO(data))
            for relative, target in (links or {}).items():
                info = tarfile.TarInfo(f"{top}/{relative}")
                info.type = tarfile.SYMTYPE
                info.linkname = target
                bundle.addfile(info)
        self.register(name, "linux-x86_64", path)

    def register(self, name, target, path):
        entry = self.manifest.setdefault(name, {"version": "1.0"})
        entry[target] = {
            "relative_path": path.relative_to(self.server).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def install(self, components, target):
        (self.server / "redistrib_99.1.0.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        dest = self.root / "cuda"
        redist.install("99.1.0", dest, components, target, self.server.as_uri(), self.root / "cache")
        return dest

    def test_windows_archives_merge_into_one_toolkit_root(self):
        self.add_zip("cuda_nvcc", {"bin/nvcc.exe": b"nvcc", "include/fatbinary_section.h": b"h"})
        self.add_zip("cuda_cudart", {"bin/x64/cudart64_99.dll": b"dll", "lib/x64/cudart.lib": b"lib"})
        dest = self.install(["cuda_nvcc", "cuda_cudart"], "windows-x86_64")
        self.assertEqual((dest / "bin" / "nvcc.exe").read_bytes(), b"nvcc")
        self.assertEqual((dest / "bin" / "x64" / "cudart64_99.dll").read_bytes(), b"dll")
        self.assertTrue((dest / "include" / "fatbinary_section.h").is_file())
        self.assertFalse((dest / "lib64").exists())
        record = json.loads((dest / "tensorsharp-cuda-redist.json").read_text(encoding="utf-8"))
        self.assertEqual(record["components"], ["cuda_nvcc", "cuda_cudart"])

    @unittest.skipIf(os.name == "nt", "symlinks need privileges on Windows")
    def test_linux_archives_keep_symlinks_and_gain_lib64(self):
        self.add_tar("cuda_cudart", {"lib/libcudart.so.99.1.0": b"so"},
                     links={"lib/libcudart.so.99": "libcudart.so.99.1.0", "lib/libcudart.so": "libcudart.so.99"})
        dest = self.install(["cuda_cudart"], "linux-x86_64")
        self.assertEqual((dest / "lib" / "libcudart.so").read_bytes(), b"so")
        self.assertTrue((dest / "lib" / "libcudart.so").is_symlink())
        self.assertEqual((dest / "lib64" / "libcudart.so.99").read_bytes(), b"so")

    def test_checksum_mismatch_is_fatal(self):
        self.add_zip("cuda_nvcc", {"bin/nvcc.exe": b"nvcc"})
        self.manifest["cuda_nvcc"]["windows-x86_64"]["sha256"] = "0" * 64
        with self.assertRaises(SystemExit) as raised:
            self.install(["cuda_nvcc"], "windows-x86_64")
        self.assertIn("SHA-256 mismatch", str(raised.exception))

    def test_missing_component_or_platform_is_fatal(self):
        self.add_zip("cuda_nvcc", {"bin/nvcc.exe": b"nvcc"})
        with self.assertRaises(SystemExit):
            self.install(["libcublas"], "windows-x86_64")
        with self.assertRaises(SystemExit):
            self.install(["cuda_nvcc"], "linux-sbsa")

    def test_cccl_installs_under_either_manifest_spelling(self):
        # CUDA 13.0.x / 13.2.x manifests call CCCL "cuda_cccl"; 13.4.x calls it "cccl".
        for listed in ("cuda_cccl", "cccl"):
            for requested in ("cccl", "cuda_cccl"):
                with self.subTest(listed=listed, requested=requested):
                    self.setUp()   # a fresh fixture server and destination per case
                    self.add_zip(listed, {"include/cuda/std/version": b"cccl"})
                    dest = self.install([requested], "windows-x86_64")
                    self.assertEqual((dest / "include" / "cuda" / "std" / "version").read_bytes(), b"cccl")
                    record = json.loads((dest / "tensorsharp-cuda-redist.json").read_text(encoding="utf-8"))
                    self.assertEqual(record["components"], [listed])

    def test_default_components_resolve_against_both_cuda13_manifest_layouts(self):
        for cccl in ("cuda_cccl", "cccl"):
            with self.subTest(cccl=cccl):
                manifest = {"release_label": "13.x"}
                for name in ("cuda_nvcc", "cuda_crt", "libnvvm", "cuda_cudart", "libcublas", "cuda_nvml_dev", cccl):
                    manifest[name] = {"linux-x86_64": {"relative_path": f"{name}/a.tar.xz", "sha256": "0" * 64}}
                resolved = redist.resolve_components(manifest, list(redist.DEFAULT_COMPONENTS), "linux-x86_64")
                self.assertEqual([name for name, _, _ in resolved][-1], cccl)
                self.assertEqual(len(resolved), len(redist.DEFAULT_COMPONENTS))

    def test_unknown_component_still_names_the_requested_spelling(self):
        with self.assertRaises(SystemExit) as raised:
            redist.resolve_components({"release_label": "13.0.3"}, ["cccl"], "linux-x86_64")
        self.assertIn("Component 'cccl' is not part of CUDA 13.0.3", str(raised.exception))

    def test_members_outside_the_root_are_refused(self):
        self.add_zip("cuda_nvcc", {"../escape.txt": b"x"})
        with self.assertRaises(SystemExit):
            self.install(["cuda_nvcc"], "windows-x86_64")
        self.assertFalse((self.root / "escape.txt").exists())


if __name__ == "__main__":
    unittest.main()
