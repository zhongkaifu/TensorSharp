"""Real CUDA Copy/SDK publish-item regression with private ASCII sentinels.

Only CopyCudaPtxToOutput and GetCopyToPublishDirectoryItems run. A late test
import replaces the nvcc detector with an empty target and removes project
references; no compiler, restore, real PTX, model, or accelerator is executed.
TENSORSHARP_CUDA_COPY_TEMP_DIR and TENSORSHARP_CUDA_COPY_EVIDENCE_DIR optionally
place fixtures and retain receipts on a chosen validation disk.
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
from xml.sax.saxutils import escape


REPO = Path(__file__).resolve().parents[3]
PROJECT = Path(os.environ.get("TENSORSHARP_CUDA_COPY_PROJECT",
    str(REPO / "TensorSharp.Backends.Cuda/TensorSharp.Backends.Cuda.csproj"))).resolve()
CONSUMER_TARGETS = os.environ.get("TENSORSHARP_CUDA_COPY_CONSUMER_TARGETS",
    str(REPO / "Directory.Build.targets"))
COMMITTED = b"committed-ptx-sentinel\n"
CORRUPTED = b"corrupted-ptx-sentinel\n"  # Same length, and optionally same timestamp.
LOCAL = b"generated-local-ptx-sentinel\n"


def contents(path):
    return {p.relative_to(path).as_posix(): p.read_bytes().decode("ascii")
            for p in path.rglob("*") if p.is_file()}


class CudaPtxCopySourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dotnet = shutil.which("dotnet")
        if cls.dotnet is None:
            raise unittest.SkipTest("Requires the real .NET SDK for Copy/publish item selection.")

    def run_fixture(self, status, *, stale_output=False, missing_local=False,
                    no_sources=False, explicit_outdir=True):
        with tempfile.TemporaryDirectory(prefix="ts-ptx copy-",
                dir=os.environ.get("TENSORSHARP_CUDA_COPY_TEMP_DIR")) as temporary:
            base = Path(temporary).resolve()
            committed, local, kernels, actual, legacy, obj = [base / name for name in
                ("committed", "generated", "kernels", "actual", "legacy", "obj")]
            for folder in (committed, local, kernels, actual, legacy, obj):
                folder.mkdir()
            intended = actual if explicit_outdir else legacy
            if not no_sources:
                (kernels / "active.cu").write_bytes(b"not CUDA code; never compiled\n")
                (committed / "active.ptx").write_bytes(COMMITTED)
                (committed / "baseline-only.ptx").write_bytes(b"committed baseline extra\n")
                if not missing_local:
                    (local / "active.ptx").write_bytes(LOCAL)
                (local / "orphan.ptx").write_bytes(b"orphan from a removed CUDA source\n")
            if stale_output:
                self.assertEqual(len(COMMITTED), len(CORRUPTED))
                source = committed / "active.ptx"
                stale = intended / "cuda_kernels/active.ptx"
                stale.parent.mkdir()
                stale.write_bytes(CORRUPTED)
                # DrvFS utime can truncate sub-second precision. Set BOTH files
                # to the same exact second, preserving the adversarial equality.
                stamp = source.stat().st_mtime_ns // 1_000_000_000 * 1_000_000_000
                os.utime(source, ns=(stamp, stamp))
                os.utime(stale, ns=(stamp, stamp))
                self.assertEqual(source.stat().st_mtime_ns, stale.stat().st_mtime_ns)
            override = base / "no-toolchain.targets"
            override.write_text('''<Project>
  <ItemGroup><ProjectReference Remove="@(ProjectReference)" /></ItemGroup>
  <Target Name="ResolveCudaPtxBuildSettings" />
</Project>''', encoding="utf-8")
            properties = dict(
                OutputPath=legacy.as_posix() + "/", BaseIntermediateOutputPath=obj.as_posix() + "/",
                MSBuildProjectExtensionsPath=obj.as_posix() + "/",
                CudaKernelDir=kernels.as_posix(), CudaCommittedPtxDir=committed.as_posix(),
                PtxOutputDir=local.as_posix(), CustomAfterMicrosoftCommonTargets=override.as_posix(),
                BuildProjectReferences="false", _GetChildProjectCopyToPublishDirectoryItems="false",
                NoBuild="true", NoRestore="true",
                UseSharedCompilation="false", TensorSharpSkipGgmlNative="true", TensorSharpSkipMlxNative="true")
            if explicit_outdir:
                properties["OutDir"] = actual.as_posix() + "/"
            if status is not None:
                properties["_NvccExitCode"] = status
            command = [self.dotnet, "msbuild", str(PROJECT), "-nologo", "-verbosity:quiet",
                "-target:CopyCudaPtxToOutput;GetCopyToPublishDirectoryItems",
                "-getTargetResult:GetCopyToPublishDirectoryItems", "-getProperty:OutDir,OutputPath,_NvccExitCode"]
            command += ["-property:" + key + "=" + value for key, value in properties.items()]
            environment = os.environ.copy()
            environment.update(DOTNET_CLI_TELEMETRY_OPTOUT="1", DOTNET_NOLOGO="1",
                CUDA_VISIBLE_DEVICES="-1", TENSORSHARP_GGML_NATIVE_SKIP="true", TENSORSHARP_MLX_NATIVE_SKIP="true")
            result = subprocess.run(command, cwd=REPO, env=environment,
                text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=60)
            receipt = dict(status=status, stale_output=stale_output, missing_local=missing_local,
                no_sources=no_sources, explicit_outdir=explicit_outdir, command=command,
                returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                actual=contents(actual), legacy=contents(legacy), committed=contents(committed),
                local=contents(local), project_sha256=hashlib.sha256(PROJECT.read_bytes()).hexdigest(),
                scope="Actual Copy target and SDK publish-item selection, fake nvcc result, no Build/restore/compiler/native/model/GPU.")
            evidence = os.environ.get("TENSORSHARP_CUDA_COPY_EVIDENCE_DIR")
            if evidence:
                destination = Path(evidence)
                destination.mkdir(parents=True, exist_ok=True)
                (destination / (uuid.uuid4().hex + ".json")).write_text(
                    json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if status == "0" and missing_local:
                self.assertNotEqual(0, result.returncode, receipt)
                self.assertIn("active.ptx", result.stdout + result.stderr)
                return
            self.assertEqual(0, result.returncode, receipt)
            output = json.loads(result.stdout)
            publish = output["TargetResults"]["GetCopyToPublishDirectoryItems"]["Items"]
            selected = {Path(item["Identity"]).name: Path(item["Identity"]).resolve()
                        for item in publish if item["Identity"].endswith(".ptx")}
            sources = {} if no_sources else ({"active.ptx": local / "active.ptx"} if status == "0" else {
                "active.ptx": committed / "active.ptx", "baseline-only.ptx": committed / "baseline-only.ptx"})
            self.assertEqual(sources, selected, receipt)
            expected = {"cuda_kernels/" + name: path.read_bytes().decode("ascii") for name, path in sources.items()}
            self.assertEqual(expected, contents(intended), receipt)
            self.assertEqual({}, contents(legacy if explicit_outdir else actual), receipt)
            self.assertEqual(intended, Path(output["Properties"]["OutDir"]).resolve())

    def test_nvcc_unavailable_uses_committed_for_copy_and_publish(self):
        self.run_fixture("9009")

    def test_unprobed_toolchain_does_not_authorize_stale_generated_ptx(self):
        self.run_fixture(None)

    def test_fallback_replaces_different_bytes_even_when_size_and_timestamp_match(self):
        self.run_fixture("9009", stale_output=True)

    def test_available_toolchain_selects_active_local_for_copy_and_publish_without_orphans(self):
        self.run_fixture("0")

    def test_available_toolchain_missing_active_output_fails_closed(self):
        self.run_fixture("0", missing_local=True)

    def test_absent_sources_copy_nothing(self):
        self.run_fixture("9009", no_sources=True)

    def test_sdk_default_outdir_uses_private_output_path(self):
        self.run_fixture("9009", explicit_outdir=False)

    def run_consumer_fixture(self, status, *, equal_timestamp=False, no_ptx=False):
        """Actual ProjectReference traversal and SDK copy/publish, no Build.

        An explicit import represents an in-repository consumer's automatic
        Directory.Build.targets import. Set CONSUMER_TARGETS=none and PROJECT to
        an exact old csproj snapshot for a genuine legacy red run. External
        consumers without this repository hook are deliberately not certified.
        """
        with tempfile.TemporaryDirectory(prefix="ts-ptx consumer-",
                dir=os.environ.get("TENSORSHARP_CUDA_COPY_TEMP_DIR")) as temporary:
            base = Path(temporary).resolve()
            committed, local, kernels, actual, publish, obj = [base / name for name in
                ("committed", "generated", "kernels", "consumer-output", "publish", "obj")]
            for folder in (committed, local, kernels, actual, publish, obj):
                folder.mkdir()
            stamp = 1_700_000_000 * 1_000_000_000  # Exact second on Windows/DrvFS.
            sources = {} if no_ptx else ({"active.ptx": local / "active.ptx"} if status == "0" else {
                "active.ptx": committed / "active.ptx", "baseline-only.ptx": committed / "baseline-only.ptx"})
            if not no_ptx:
                (kernels / "active.cu").write_bytes(b"not CUDA code; never compiled\n")
                (committed / "active.ptx").write_bytes(COMMITTED)
                (committed / "baseline-only.ptx").write_bytes(b"committed baseline extra\n")
                (local / "active.ptx").write_bytes(LOCAL)
                (local / "orphan.ptx").write_bytes(b"orphan is not an active producer\n")
                selected = sources["active.ptx"]
                desired = selected.read_bytes()
                previous = b"X" * len(desired)
                # Change source content in place without changing its size or
                # timestamp, as can happen after restoring a producer snapshot.
                selected.write_bytes(previous)
                os.utime(selected, ns=(stamp, stamp))
                for destination in (actual, publish):
                    stale = destination / "cuda_kernels/active.ptx"
                    stale.parent.mkdir(parents=True)
                    stale.write_bytes(previous)
                    target_stamp = stamp if equal_timestamp else stamp + 60 * 1_000_000_000
                    os.utime(stale, ns=(target_stamp, target_stamp))
                selected.write_bytes(desired)
                os.utime(selected, ns=(stamp, stamp))
                self.assertEqual(len(previous), len(desired))

            # Same-metadata controls would be skipped by SDK publish. The narrow
            # hook must not force-copy non-PTX, out-of-scope or Never content.
            control_rows = [
                ("ordinary.txt", "cuda_kernels/ordinary.txt", "Always"),
                ("foreign.ptx", "other_kernels/foreign.ptx", "Always"),
                ("never.ptx", "cuda_kernels/never.ptx", "Never"),
                ("escape.ptx", "cuda_kernels/../escape.ptx", "Always")]
            controls = []
            for filename, target, policy in control_rows:
                source = base / filename
                source.write_bytes(b"new-payload\n")
                os.utime(source, ns=(stamp, stamp))
                destination = (publish / target).resolve()
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"old-payload\n")
                os.utime(destination, ns=(stamp, stamp))
                controls.append((source, target, policy, destination))

            override = base / "no-build.targets"
            override.write_text('''<Project>
  <ItemGroup Condition="'$(MSBuildProjectName)' == 'TensorSharp.Backends.Cuda'">
    <ProjectReference Remove="@(ProjectReference)" />
  </ItemGroup>
  <Target Name="ResolveCudaPtxBuildSettings" />
  <Target Name="CompileCudaPtx"><Error Text="Fixture must never compile PTX." /></Target>
  <Target Name="CoreCompile"><Error Text="Fixture must never compile managed code." /></Target>
</Project>''', encoding="utf-8")
            consumer = base / "Consumer.csproj"
            hook = "" if CONSUMER_TARGETS == "none" else (
                '<Import Project="' + escape(str(Path(CONSUMER_TARGETS).resolve()), {'"': '&quot;'}) + '" />')
            content = "\n".join('<Content Include="' + escape(str(source), {'"': '&quot;'})
                + '" Link="' + escape(target, {'"': '&quot;'})
                + '" CopyToOutputDirectory="Never" CopyToPublishDirectory="' + policy + '" />'
                for source, target, policy, _ in controls)
            consumer.write_text('''<Project>
  <Import Project="Sdk.props" Sdk="Microsoft.NET.Sdk" />
  <PropertyGroup><TargetFramework>net10.0</TargetFramework><EnableDefaultItems>false</EnableDefaultItems>
    <AppendTargetFrameworkToOutputPath>false</AppendTargetFrameworkToOutputPath>
    <OutputPath>''' + escape(actual.as_posix() + "/") + '''</OutputPath>
    <PublishDir>''' + escape(publish.as_posix() + "/") + '''</PublishDir>
  </PropertyGroup>
  <ItemGroup><ProjectReference Include="''' + escape(str(PROJECT), {'"': '&quot;'}) + '''"
      ReferenceOutputAssembly="false" SkipGetTargetFrameworkProperties="true" />
''' + content + '''</ItemGroup>
''' + hook + '''
  <Target Name="CopySentinels" DependsOnTargets="_CopySourceItemsToOutputDirectory;_ComputeCopyToPublishDirectoryItems;CopyFilesToPublishDirectory" />
  <Import Project="Sdk.targets" Sdk="Microsoft.NET.Sdk" />
</Project>''', encoding="utf-8")
            properties = dict(BaseIntermediateOutputPath=obj.as_posix() + "/",
                MSBuildProjectExtensionsPath=obj.as_posix() + "/",
                CudaKernelDir=kernels.as_posix(), CudaCommittedPtxDir=committed.as_posix(),
                PtxOutputDir=local.as_posix(), AfterMicrosoftNETSdkTargets=override.as_posix(),
                BuildProjectReferences="false", NoBuild="true", NoRestore="true",
                SkipResolvePackageAssets="true",
                _GetChildProjectCopyToOutputDirectoryItems="true", _GetChildProjectCopyToPublishDirectoryItems="true",
                UseSharedCompilation="false", TensorSharpSkipGgmlNative="true", TensorSharpSkipMlxNative="true",
                _NvccExitCode=status)
            command = [self.dotnet, "msbuild", str(consumer), "-nologo", "-verbosity:quiet",
                "-target:CopySentinels"]
            command += ["-property:" + key + "=" + value for key, value in properties.items()]
            environment = os.environ.copy()
            environment.update(DOTNET_CLI_TELEMETRY_OPTOUT="1", DOTNET_NOLOGO="1", CUDA_VISIBLE_DEVICES="-1")
            result = subprocess.run(command, cwd=REPO, env=environment, text=True,
                encoding="utf-8", errors="replace", capture_output=True, timeout=60)
            receipt = dict(status=status, equal_timestamp=equal_timestamp, no_ptx=no_ptx,
                command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                actual=contents(actual), publish=contents(publish), source=contents(local if status == "0" else committed),
                project=str(PROJECT), project_sha256=hashlib.sha256(PROJECT.read_bytes()).hexdigest(),
                consumer_targets=CONSUMER_TARGETS,
                scope="Real SDK ProjectReference content copy/publish; no Build/compiler/restore/nvcc/model/GPU.")
            evidence = os.environ.get("TENSORSHARP_CUDA_COPY_EVIDENCE_DIR")
            if evidence:
                destination = Path(evidence); destination.mkdir(parents=True, exist_ok=True)
                (destination / (uuid.uuid4().hex + ".json")).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            self.assertEqual(0, result.returncode, receipt)
            for name, source in sources.items():
                self.assertEqual(source.read_bytes(), (actual / "cuda_kernels" / name).read_bytes(), receipt)
                self.assertEqual(source.read_bytes(), (publish / "cuda_kernels" / name).read_bytes(), receipt)
            self.assertEqual(set("cuda_kernels/" + name for name in sources), set(contents(actual)), receipt)
            for _, _, _, destination in controls:
                self.assertEqual(b"old-payload\n", destination.read_bytes(), receipt)

    def test_consumer_fallback_overwrites_newer_previous_producer_and_publishes(self):
        self.run_consumer_fixture("9009")

    def test_consumer_fallback_same_metadata_content_change_is_not_skipped(self):
        self.run_consumer_fixture("9009", equal_timestamp=True)

    def test_consumer_local_overwrites_newer_previous_producer_and_publishes(self):
        self.run_consumer_fixture("0")

    def test_consumer_local_same_metadata_content_change_is_not_skipped(self):
        self.run_consumer_fixture("0", equal_timestamp=True)

    def test_consumer_without_cuda_payload_keeps_unrelated_publish_files(self):
        self.run_consumer_fixture("9009", no_ptx=True)


if __name__ == "__main__":
    unittest.main()
