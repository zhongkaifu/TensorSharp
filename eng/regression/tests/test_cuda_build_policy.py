"""Exercise actual PTX target XML without an SDK, compiler, GPU, or model build."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[3] / "TensorSharp.Backends.Cuda/TensorSharp.Backends.Cuda.csproj"
DIRECTORY_TARGETS = PROJECT.parent.parent / "Directory.Build.targets"
GUARDED_TARGETS = ("DetectCudaPtxBuildSettings", "ResolveCudaPtxArchitecture", "ResolveCudaPtxBuildSettings",
                   "SelectCudaPtxContent", "CompileCudaPtx", "CopyCudaPtxToOutput", "UpdateCommittedCudaPtx")


class CudaBuildPolicySourceTests(unittest.TestCase):
    def test_native_targets_and_baseline_content_have_explicit_skip_gates(self):
        project = ET.parse(PROJECT).getroot()
        targets = {target.attrib["Name"]: target for target in project.findall("Target")}
        for name in GUARDED_TARGETS:
            self.assertIn("'$(TensorSharpSkipCudaPtx)' != 'true'", targets[name].get("Condition", ""), name)
        baseline = next(item for item in project.findall("./ItemGroup/Content") if "CudaCommittedPtxDir" in item.get("Include", ""))
        self.assertIn("'$(TensorSharpSkipCudaPtx)' != 'true'", baseline.get("Condition", ""))
        validation = targets["ValidatePrebuiltCudaPtx"]
        self.assertIn("'$(TensorSharpSkipCudaPtx)' == 'true'", validation.get("Condition", ""))
        self.assertIn("BeforeBuild", validation.get("BeforeTargets", ""))
        self.assertIn("AssignTargetPaths", validation.get("BeforeTargets", ""))
        self.assertTrue(validation.findall("Error"))
        self.assertEqual([], validation.findall("Exec"))
        self.assertEqual([], validation.findall("Copy"))
        shared = ET.parse(DIRECTORY_TARGETS).getroot()
        retention = next(item for item in shared.findall("Target") if item.attrib["Name"] == "TensorSharpPreservePrebuiltCudaAssets")
        self.assertIn("_CleanGetCurrentAndPriorFileWrites", retention.get("BeforeTargets", ""))
        self.assertIn("'$(TensorSharpSkipCudaPtx)' == 'true'", retention.get("Condition", ""))
        self.assertTrue(retention.findall("./ItemGroup/FileWrites"))


@unittest.skipUnless(shutil.which("dotnet"), "MSBuild XML contract requires local dotnet; no native build is performed")
class CudaBuildPolicyTargetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.active = self.root / "output" / "cuda_kernels"
        self.active.mkdir(parents=True)
        self.ptx = self.active / "sample.ptx"
        self.ptx.write_text(".version 9.0\n.target sm_75\n.address_size 64\n.visible .entry sample() { ret; }\n")
        self.fatbin = self.active / "sample.fatbin"
        self.fatbin.write_bytes(b"prebuilt-fatbin-untouched")
        self.sidecar = self.active / "sample.fatbin.sha256"
        self.sidecar.write_text("prebuilt-sidecar-untouched")
        kernel_dir = self.root / "native/kernels"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "sample.cu").write_text("no compiler may read this source")
        baseline = self.root / "native/ptx"
        baseline.mkdir()
        (baseline / "sample.ptx").write_text("committed fallback must not replace active output")
        intermediate = self.root / "obj/cuda_ptx/ptx"
        intermediate.mkdir(parents=True)
        (intermediate / "sample.ptx").write_text("stale intermediate must not replace active output")
        # Preserve the actual property/item/target definitions, but remove SDK
        # and project references: only explicit MSBuild targets can execute.
        project = ET.parse(PROJECT).getroot()
        project.attrib.pop("Sdk", None)
        for group in project.findall("ItemGroup"):
            for reference in list(group.findall("ProjectReference")):
                group.remove(reference)
        properties = ET.Element("PropertyGroup")
        for key, value in {"TensorSharpSkipCudaPtx": "true", "OutDir": str(self.root / "output"),
                           "CudaNvcc": str(self.root / "must-not-run-nvcc"), "_NvccExitCode": "0",
                           "_CudaFatbinGencode": "not-compiled"}.items():
            ET.SubElement(properties, key).text = value
        project.insert(0, properties)
        for shared_target in ET.parse(DIRECTORY_TARGETS).getroot():
            project.append(shared_target)
        target = ET.SubElement(project, "Target", Name="AssertNoFallbackContent")
        ET.SubElement(target, "Error", Condition="'@(Content)' != ''", Text="PTX content selection or copying remains enabled")
        ET.SubElement(project, "Target", Name="CheckManagedOnlyPolicy",
                      DependsOnTargets=";".join(("ValidatePrebuiltCudaPtx", *GUARDED_TARGETS, "AssertNoFallbackContent")))
        # Model the SDK's orphan calculation against prior FileWrites. The real
        # repository retention target must run before current writes are read.
        current = ET.SubElement(project, "Target", Name="_CleanGetCurrentAndPriorFileWrites")
        items = ET.SubElement(current, "ItemGroup")
        ET.SubElement(items, "_PriorWrites", Include=str(self.active / "*"))
        ET.SubElement(items, "_CurrentWrites", Include="@(FileWrites)")
        cleanup = ET.SubElement(project, "Target", Name="SimulateIncrementalClean", DependsOnTargets="_CleanGetCurrentAndPriorFileWrites")
        items = ET.SubElement(cleanup, "ItemGroup")
        ET.SubElement(items, "_OrphanWrites", Include="@(_PriorWrites)", Exclude="@(_CurrentWrites)")
        ET.SubElement(cleanup, "Delete", Files="@(_OrphanWrites)")
        for name in ("Clean", "CoreClean", "Rebuild"):
            ET.SubElement(project, "Target", Name=name)
        self.project = self.root / "policy.proj"
        ET.ElementTree(project).write(self.project, encoding="utf-8", xml_declaration=True)

    def invoke(self, *properties, target="CheckManagedOnlyPolicy"):
        return subprocess.run([shutil.which("dotnet"), "msbuild", str(self.project), "-nologo", "-verbosity:quiet",
                               "-target:" + target, "-maxcpucount:1", "-nodeReuse:false", *properties],
                              cwd=self.root, text=True, capture_output=True, timeout=30, check=False)

    def test_skip_neither_probes_nvcc_selects_baseline_nor_mutates_active_assets(self):
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (self.ptx, self.fatbin, self.sidecar)}
        result = self.invoke()
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertNotIn("must-not-run-nvcc", result.stdout + result.stderr)
        self.assertEqual(before, {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before})

    def test_missing_prebuilt_fails_even_when_committed_and_intermediate_files_exist(self):
        self.ptx.unlink()
        result = self.invoke()
        self.assertNotEqual(0, result.returncode)
        self.assertIn("requires prebuilt CUDA PTX", result.stdout + result.stderr)
        self.assertFalse(self.ptx.exists())

    def test_invalid_prebuilt_ptx_fails_without_replacement(self):
        for text in ("", ".version 9.0\n.target invalid\n"):
            self.ptx.write_text(text)
            result = self.invoke()
            self.assertNotEqual(0, result.returncode)
            self.assertIn("Prebuilt CUDA PTX has no valid", result.stdout + result.stderr)
            self.assertEqual(text, self.ptx.read_text())

    def test_committed_refresh_is_incompatible_with_managed_only_flag(self):
        result = self.invoke("-property:TensorSharpUpdateCommittedPtx=true")
        self.assertNotEqual(0, result.returncode)
        self.assertIn("cannot refresh committed kernels", result.stdout + result.stderr)

    def test_referencing_project_incremental_cleanup_retains_existing_cuda_assets(self):
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (self.ptx, self.fatbin, self.sidecar)}
        result = self.invoke(target="SimulateIncrementalClean")
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertEqual(before, {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before})

    def test_clean_and_rebuild_fail_before_removing_prebuilt_assets(self):
        for target in ("Clean", "CoreClean", "Rebuild"):
            result = self.invoke(target=target)
            self.assertNotEqual(0, result.returncode)
            self.assertIn("cannot be combined with Clean or Rebuild", result.stdout + result.stderr)
            self.assertTrue(self.ptx.is_file())


if __name__ == "__main__":
    unittest.main()
