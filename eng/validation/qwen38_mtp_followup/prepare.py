#!/usr/bin/env python3
"""Prepare an isolated Qwen MTP benchmark from pinned, already-frozen source; no build/run."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_one(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Frozen benchmark seam changed: ' + old[:100])
    return text.replace(old, new)


def prepare(package, expected, output):
    raw = (package / 'manifest.json').read_bytes()
    if sha(raw) != expected:
        raise ValueError('Frozen source package manifest changed')
    manifest = json.loads(raw)
    archive = (package / 'source.tar.gz').read_bytes()
    if sha(archive) != manifest['archive_sha256']:
        raise ValueError('Frozen source archive changed')
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    prefix = 'benchmarks/AgentTurnBench/'
    with tarfile.open(package / 'source.tar.gz') as stream:
        for member in stream:
            if member.name.startswith(prefix) and member.name.endswith('.cs'):
                data = stream.extractfile(member).read()
                if sha(data) != manifest['files'][member.name]['sha256']:
                    raise ValueError('Frozen benchmark input changed: ' + member.name)
                name = Path(member.name).name
                if name in sources:
                    raise ValueError('Duplicate source basename')
                sources[name] = {'source': member.name, 'original_sha256': sha(data)}
                if name == 'Program.cs':
                    text = data.decode('utf-8-sig')
                    text = replace_one(text, 'internal sealed class Bench', 'internal sealed partial class Bench')
                    text = replace_one(text, 'case "short": await ShortAsync(); break;',
                                       'case "qwen-mtp": await QwenMtpAsync(); break;\n                    case "short": await ShortAsync(); break;')
                    text = replace_one(text, 'model.WarmUpKernels();', 'Bench.RequireMtpHead(model, o);\n        model.WarmUpKernels();')
                    data = text.encode('utf-8')
                (output / name).write_bytes(data)
                sources[name]['derived_sha256'] = sha(data)
    for name in ('Program.cs', 'ConcurrentDecodeMetrics.cs', 'SpecParityDiagnostic.cs'):
        if name not in sources:
            raise ValueError('Missing required frozen benchmark source: ' + name)
    extension = Path(__file__).with_name('QwenMtpScenarios.cs').read_bytes()
    (output / 'QwenMtpScenarios.cs').write_bytes(extension)
    # References point at the future pinned managed app; building this project
    # cannot trigger a production ProjectReference or native build.
    project = ET.Element('Project', Sdk='Microsoft.NET.Sdk')
    prop = ET.SubElement(project, 'PropertyGroup')
    for key, value in {'OutputType': 'Exe', 'TargetFramework': 'net10.0', 'ImplicitUsings': 'enable',
                       'Nullable': 'disable', 'AssemblyName': 'QwenMtpFollowup', 'EnableDefaultCompileItems': 'true'}.items():
        ET.SubElement(prop, key).text = value
    group = ET.SubElement(project, 'ItemGroup')
    ET.SubElement(group, 'FrameworkReference', Include='Microsoft.AspNetCore.App')
    for name in ('AdvUtils', 'TensorSharp.Core', 'TensorSharp.Models', 'TensorSharp.Runtime',
                 'TensorSharp.Runtime.Logging', 'TensorSharp.Backends.GGML', 'TensorSharp.Backends.Cuda',
                 'TensorSharp.Backends.MLX', 'TensorSharp.Distributed'):
        ref = ET.SubElement(group, 'Reference', Include=name)
        ET.SubElement(ref, 'HintPath').text = '$(FrozenManagedDir)/' + name + '.dll'
        ET.SubElement(ref, 'Private').text = 'true'
    ET.indent(project)
    (output / 'QwenMtpFollowup.csproj').write_text(ET.tostring(project, encoding='unicode'), encoding='utf-8')
    result = {'schema': 1, 'status': 'prepared-not-built-not-run', 'release_qualified': False,
              'source_package_manifest_sha256': expected, 'source_archive_sha256': manifest['archive_sha256'],
              'source_package': str(package), 'frozen_sources': sources,
              'required_future_inputs': ['built native SHA256', 'complete managed application file manifest SHA256',
                                         'build result SHA256 retaining all original failures', 'head/trunk/mmproj/fixture identity'],
              'files': {p.name: sha(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / 'prepared.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.package, args.manifest_sha256, args.output)
    print(json.dumps({'status': result['status'], 'files': len(result['files'])}))


if __name__ == '__main__':
    main()
