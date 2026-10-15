"""Execute the real SDK native-copy targets with tiny fake files, never Build.

No model, native toolchain, restore, or native binary is required. OutputPath,
OutDir and both native source directories always point into a private fixture.
TargetDir is deliberately not supplied: the .NET SDK must derive it from OutDir.
Optional TENSORSHARP_NATIVE_COPY_EVIDENCE_DIR retains each invocation receipt;
TENSORSHARP_NATIVE_COPY_TEMP_DIR places temporary fixtures on a chosen disk.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import uuid


REPO = Path(__file__).resolve().parents[3]
TARGETS = (
    ("TensorSharp.Server.Host/TensorSharp.Server.Host.csproj", "CopyGgmlNativeBinary"),
    ("TensorSharp.Backends.GGML/TensorSharp.Backends.GGML.csproj", "CopyGgmlNativeBinary"),
    ("TensorSharp.Server.Host/TensorSharp.Server.Host.csproj", "CopyMlxNativeBinary"),
    ("TensorSharp.Backends.MLX/TensorSharp.Backends.MLX.csproj", "CopyMlxNativeBinary"),
)
NAMES = dict(GgmlNativeBinaryName="fake-ggml.sentinel",
             MlxNativeBinaryName="fake-mlxc.sentinel",
             MlxNativeDependencyName="fake-mlx.sentinel",
             MlxNativeJacclName="fake-jaccl.sentinel",
             MlxNativeMetallibName="fake-metal.sentinel")


def files(folder):
    return {str(p.relative_to(folder)).replace("\\", "/"): p.read_bytes().decode("ascii")
            for p in folder.rglob("*") if p.is_file()}


class NativeCopyOutDirTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dotnet = shutil.which("dotnet")
        if cls.dotnet is None:
            raise unittest.SkipTest("Real native-copy target regression requires the .NET SDK.")

    def run_target(self, project, target, layout="root", ios=False,
                   missing=False, explicit_outdir=True):
        with tempfile.TemporaryDirectory(prefix="ts-native copy-",
                dir=os.environ.get("TENSORSHARP_NATIVE_COPY_TEMP_DIR")) as temporary:
            base = Path(temporary).resolve()
            native = base / "native"
            actual = base / "actual"
            legacy = base / "legacy"
            obj = base / "obj"
            for folder in (native, actual, legacy, obj):
                folder.mkdir()
            expected = {}
            if not missing:
                if target == "CopyGgmlNativeBinary":
                    folder = native / "Release" if layout == "nested" else native
                    folder.mkdir(exist_ok=True)
                    name = NAMES["GgmlNativeBinaryName"]
                    expected[name] = "fake GGML payload; not a native binary\n"
                    (folder / name).write_bytes(expected[name].encode("ascii"))
                else:
                    for key in ("MlxNativeBinaryName", "MlxNativeDependencyName",
                                "MlxNativeJacclName", "MlxNativeMetallibName"):
                        folder = native / "lib" if layout == "nested" and key in (
                            "MlxNativeJacclName", "MlxNativeMetallibName") else native
                        folder.mkdir(exist_ok=True)
                        name = NAMES[key]
                        expected[name] = "fake " + key + "; not a native binary\n"
                        (folder / name).write_bytes(expected[name].encode("ascii"))
                    if layout == "root":
                        alternative = native / "lib" / NAMES["MlxNativeMetallibName"]
                        alternative.parent.mkdir(exist_ok=True)
                        alternative.write_bytes(b"unused nested metallib alternative\n")
            properties = dict(NAMES, OutputPath=legacy.as_posix() + "/",
                BaseIntermediateOutputPath=obj.as_posix() + "/",
                MSBuildProjectExtensionsPath=obj.as_posix() + "/",
                GgmlNativeBuildDir=native.as_posix(), MlxNativeInstallDir=native.as_posix(),
                TensorSharpSkipGgmlNative="true", TensorSharpSkipMlxNative="true")
            if explicit_outdir:
                properties["OutDir"] = actual.as_posix() + "/"
            if ios:
                properties["TargetPlatformIdentifier"] = "ios"
            self.assertNotIn("TargetDir", properties)
            command = [self.dotnet, "msbuild", str(REPO / project), "-nologo",
                "-verbosity:quiet", "-target:" + target,
                "-getProperty:OutputPath,OutDir,TargetDir"]
            command += ["-property:" + key + "=" + value for key, value in properties.items()]
            environment = os.environ.copy()
            environment.update(DOTNET_CLI_TELEMETRY_OPTOUT="1", DOTNET_NOLOGO="1",
                DOTNET_SKIP_FIRST_TIME_EXPERIENCE="1", CUDA_VISIBLE_DEVICES="-1",
                TENSORSHARP_GGML_NATIVE_SKIP="true", TENSORSHARP_MLX_NATIVE_SKIP="true")
            result = subprocess.run(command, cwd=REPO, env=environment,
                text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=90)
            receipt = dict(project=project, target=target, layout=layout, ios=ios,
                missing=missing, explicit_outdir=explicit_outdir, command=command,
                returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                actual=files(actual), legacy=files(legacy), source=files(native),
                project_sha256=hashlib.sha256((REPO / project).read_bytes()).hexdigest(),
                scope="Only a real SDK Copy target with private ASCII sentinels; no Build/restore/native/GPU execution.")
            evidence = os.environ.get("TENSORSHARP_NATIVE_COPY_EVIDENCE_DIR")
            if evidence:
                destination = Path(evidence)
                destination.mkdir(parents=True, exist_ok=True)
                (destination / (uuid.uuid4().hex + ".json")).write_text(
                    json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.assertEqual(0, result.returncode, json.dumps(receipt, ensure_ascii=False))
            derived = json.loads(result.stdout)["Properties"]
            intended = actual if explicit_outdir else legacy
            self.assertEqual(intended, Path(derived["TargetDir"]).resolve())
            self.assertEqual(intended, Path(derived["OutDir"]).resolve())
            if ios:
                self.assertEqual({}, receipt["actual"])
                self.assertEqual({}, receipt["legacy"])
            else:
                self.assertEqual(expected, files(intended), json.dumps(receipt, ensure_ascii=False))
                if explicit_outdir:
                    self.assertEqual({}, receipt["legacy"], json.dumps(receipt, ensure_ascii=False))

    def test_explicit_outdir_uses_real_sdk_target_directory(self):
        for project, target in TARGETS:
            with self.subTest(project=project, target=target):
                self.run_target(project, target)

    def test_release_and_nested_mlx_layouts_follow_outdir(self):
        for project, target in TARGETS:
            with self.subTest(project=project, target=target):
                self.run_target(project, target, layout="nested")

    def test_backend_ios_copy_exclusions_remain_in_force(self):
        for project, target in (TARGETS[1], TARGETS[3]):
            with self.subTest(project=project, target=target):
                self.run_target(project, target, ios=True)

    def test_missing_sources_do_not_copy_files(self):
        for project, target in TARGETS:
            with self.subTest(project=project, target=target):
                self.run_target(project, target, missing=True)

    def test_default_outdir_still_uses_sdk_output_path(self):
        for project, target in TARGETS:
            with self.subTest(project=project, target=target):
                self.run_target(project, target, explicit_outdir=False)


if __name__ == "__main__":
    unittest.main()
