"""Execute Probe's deployment MSBuild target using synthetic libraries only."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[2] / "ForcedLogitProbe/ForcedLogitProbe.csproj"


class ProbeDeploymentSourceTests(unittest.TestCase):
    def test_copy_target_has_no_native_build_step_and_keeps_model_free_contract_excluded(self):
        project = ET.parse(PROJECT).getroot()
        target = next(item for item in project.findall("Target") if item.get("Name") == "CopyGgmlNativeBinary")
        self.assertEqual("Build", target.get("AfterTargets"))
        self.assertFalse(target.findall("Exec"))
        self.assertFalse(target.findall("MSBuild"))
        self.assertFalse(target.findall("CallTarget"))
        self.assertTrue(target.findall("Warning"))
        self.assertTrue(target.findall("Error"))
        self.assertEqual("ContractTests/**/*.cs", project.find("./ItemGroup/Compile").get("Remove"))
        defaults = project.findall("./PropertyGroup/GgmlNativeBuildDir")
        self.assertTrue(all("'$(GgmlNativeBuildDir)' == ''" in item.get("Condition", "") for item in defaults))


@unittest.skipUnless(shutil.which("dotnet"), "Deployment target contract requires local MSBuild; no compilation is performed")
class ProbeDeploymentTargetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.native = self.root / "isolated native build"
        self.output = self.root / "deployed probe"
        self.native.mkdir()
        # Copy the actual target/properties, removing SDK/project references so
        # Build is an empty hook and cannot launch a compiler or native build.
        project = ET.parse(PROJECT).getroot()
        project.attrib.pop("Sdk", None)
        for group in project.findall("ItemGroup"):
            for reference in list(group.findall("ProjectReference")):
                group.remove(reference)
        ET.SubElement(project, "Target", Name="Build")
        self.project = self.root / "deployment.proj"
        ET.ElementTree(project).write(self.project, encoding="utf-8", xml_declaration=True)

    def invoke(self, name="GgmlOps.dll", *properties):
        return subprocess.run([shutil.which("dotnet"), "msbuild", str(self.project), "-nologo", "-verbosity:minimal",
                               "-target:Build", "-maxcpucount:1", "-nodeReuse:false",
                               "-property:GgmlNativeBuildDir=" + str(self.native),
                               "-property:GgmlNativeBinaryName=" + name,
                               "-property:OutDir=" + str(self.output), *properties],
                              cwd=self.root, text=True, capture_output=True, timeout=30, check=False)

    def test_prebuilt_root_layout_copies_platform_binary_without_compiling(self):
        for name in ("GgmlOps.dll", "libGgmlOps.so", "libGgmlOps.dylib"):
            with self.subTest(name=name):
                (self.native / name).write_bytes(b"synthetic prebuilt " + name.encode())
                result = self.invoke(name)
                self.assertEqual(0, result.returncode, result.stdout + result.stderr)
                self.assertEqual((self.native / name).read_bytes(), (self.output / name).read_bytes())
                self.assertIn("prebuilt GGML runtime:", result.stdout)

    def test_release_layout_wins_once_when_both_sources_exist(self):
        (self.native / "GgmlOps.dll").write_bytes(b"root prebuilt")
        release = self.native / "Release"
        release.mkdir()
        (release / "GgmlOps.dll").write_bytes(b"Release prebuilt")
        result = self.invoke()
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertEqual(b"Release prebuilt", (self.output / "GgmlOps.dll").read_bytes())
        self.assertEqual(1, result.stdout.count("prebuilt GGML runtime:"))

    def test_resolved_target_directory_takes_precedence_over_outdir(self):
        (self.native / "GgmlOps.dll").write_bytes(b"prebuilt runtime")
        target = self.root / "actual framework output"
        result = self.invoke("GgmlOps.dll", "-property:TargetDir=" + str(target))
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertEqual(b"prebuilt runtime", (target / "GgmlOps.dll").read_bytes())
        self.assertFalse(self.output.exists())

    def test_missing_prebuilt_warns_and_never_labels_stale_output_as_copied(self):
        self.output.mkdir()
        stale = self.output / "GgmlOps.dll"
        stale.write_bytes(b"existing explicit deployment")
        result = self.invoke()
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertIn("GGML deployment unavailable", result.stdout + result.stderr)
        self.assertIn("existing output library is unchanged", result.stdout + result.stderr)
        self.assertNotIn("prebuilt GGML runtime:", result.stdout + result.stderr)
        self.assertEqual(b"existing explicit deployment", stale.read_bytes())

    def test_required_deployment_fails_without_implicit_native_build(self):
        result = self.invoke("GgmlOps.dll", "-property:ForcedLogitProbeRequireGgmlNative=true")
        self.assertNotEqual(0, result.returncode)
        self.assertIn("No native build was attempted", result.stdout + result.stderr)
        self.assertFalse(self.output.exists())
        self.assertEqual([], list(self.native.iterdir()))


if __name__ == "__main__":
    unittest.main()
