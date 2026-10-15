#!/usr/bin/env python3
"""Strict CPU F32 Qwen-compatible vision-component reference, never a full LM.

prepare is light: stdlib, bounded GGUF header, source/package metadata only.
run is explicitly authorized separately and executes a bounded child. No model
download, dependency installation, package modification, or GPU execution exists.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CONFIG = {'depth': 27, 'hidden_size': 1152, 'hidden_act': 'gelu_pytorch_tanh',
          'intermediate_size': 4304, 'num_heads': 16, 'in_channels': 3,
          'patch_size': 16, 'spatial_merge_size': 2, 'temporal_patch_size': 2,
          'out_hidden_size': 2560, 'num_position_embeddings': 2304,
          'deepstack_visual_indexes': [], 'initializer_range': 0.02}
IMAGE_SHA = '10145e33904f59e8f4e61cdd035cebbf45d7e5591d8ded6cdedd5aa82e022b18'


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def pin(path):
    path = Path(path).resolve()
    return {'Path': str(path), 'Bytes': path.stat().st_size, 'Sha256': sha(path)}


def metadata(path):
    path = Path(path).resolve(); stat = path.stat()
    return {'Path': str(path), 'Bytes': stat.st_size, 'MtimeNs': stat.st_mtime_ns}


def fresh(path):
    path = Path(path).resolve()
    if path.exists():
        raise ValueError('Fresh output directory required: ' + str(path))
    path.mkdir(parents=True); return path


def require(value, message):
    if not value:
        raise ValueError(message)


def directory_mapping(config):
    """Explicit complete GGUF->HF key map. Two temporal slices share one target."""
    h, ff, p = config['hidden_size'], config['intermediate_size'], config['patch_size']
    merger = h * config['spatial_merge_size'] ** 2
    result = []
    def add(source, destination, shape, temporal_slice=None):
        row = {'Source': source, 'Destination': destination, 'SourceShape': shape}
        if temporal_slice is not None:
            row['TemporalSlice'] = temporal_slice
        result.append(row)
    add('v.patch_embd.weight', 'patch_embed.proj.weight', [h,3,p,p], 0)
    add('v.patch_embd.weight.1', 'patch_embed.proj.weight', [h,3,p,p], 1)
    add('v.patch_embd.bias', 'patch_embed.proj.bias', [h])
    add('v.position_embd.weight', 'pos_embed.weight', [config['num_position_embeddings'],h])
    add('v.post_ln.weight', 'merger.norm.weight', [h])
    add('v.post_ln.bias', 'merger.norm.bias', [h])
    for index in range(config['depth']):
        for source, destination, shape in [
                ('ln1.weight','norm1.weight',[h]),('ln1.bias','norm1.bias',[h]),
                ('ln2.weight','norm2.weight',[h]),('ln2.bias','norm2.bias',[h]),
                ('attn_qkv.weight','attn.qkv.weight',[3*h,h]),('attn_qkv.bias','attn.qkv.bias',[3*h]),
                ('attn_out.weight','attn.proj.weight',[h,h]),('attn_out.bias','attn.proj.bias',[h]),
                ('ffn_up.weight','mlp.linear_fc1.weight',[ff,h]),('ffn_up.bias','mlp.linear_fc1.bias',[ff]),
                ('ffn_down.weight','mlp.linear_fc2.weight',[h,ff]),('ffn_down.bias','mlp.linear_fc2.bias',[h])]:
            add(f'v.blk.{index}.{source}', f'blocks.{index}.{destination}', shape)
    add('mm.0.weight','merger.linear_fc1.weight',[merger,merger])
    add('mm.0.bias','merger.linear_fc1.bias',[merger])
    add('mm.2.weight','merger.linear_fc2.weight',[config['out_hidden_size'],merger])
    add('mm.2.bias','merger.linear_fc2.bias',[config['out_hidden_size']])
    return result


def validate_directory(tensors, mapping):
    by_name = {row['name']: row for row in tensors}
    require(len(by_name) == len(tensors), 'Duplicate GGUF tensor name')
    require(set(by_name) == {row['Source'] for row in mapping}, 'Missing/unexpected GGUF tensor; no weight may be dropped')
    for row in mapping:
        tensor = by_name[row['Source']]
        require(list(reversed(tensor['dimensions'])) == row['SourceShape'], 'GGUF shape mismatch: ' + row['Source'])
        require(tensor['type'] in (0,30), 'Only actual F32/BF16 checkpoint tensors are supported: ' + row['Source'])
        require(tensor['stored_bytes'] == math.prod(row['SourceShape']) * (4 if tensor['type'] == 0 else 2),
                'GGUF payload byte count mismatch: ' + row['Source'])
    return by_name


def header(path):
    spec = importlib.util.spec_from_file_location('qwen_hf_header', HERE / 'inspect-remote-gguf.py')
    reader = importlib.util.module_from_spec(spec); spec.loader.exec_module(reader)
    with Path(path).open('rb') as stream:
        data = stream.read(1 << 20)
    result = reader.inspect(data, Path(path).stat().st_size)
    require(result['header_bytes'] <= len(data), 'Bounded GGUF header exceeded 1 MiB')
    return result, hashlib.sha256(data[:result['header_bytes']]).hexdigest()


def package_file(package, relative):
    spec = importlib.util.find_spec(package)
    require(spec is not None and spec.origin is not None, 'Required installed package absent: ' + package)
    path = Path(spec.origin).parent / relative
    require(path.is_file(), 'Required installed reference source absent: ' + str(path))
    return path


def prepare(args):
    out = fresh(args.output); model = Path(args.mmproj).resolve(); image = Path(args.image).resolve()
    require(sha(image) == IMAGE_SHA, 'Require exact opaque640x480 card-0 image')
    # Parse image geometry from a small PNG header without PIL/torch imports.
    raw = image.read_bytes()
    require(raw[:8] == b'\x89PNG\r\n\x1a\n', 'Require original PNG fixture')
    import struct
    require(struct.unpack('>II',raw[16:24]) == (640,480), 'Require exact640x480 no-resize fixture')
    data, header_sha = header(model); meta = data['metadata']
    expected_meta = {'general.architecture':'clip','clip.projector_type':'qwen3vl_merger',
                     'clip.vision.patch_size':16,'clip.vision.embedding_length':1152,
                     'clip.vision.feed_forward_length':4304,'clip.vision.attention.head_count':16,
                     'clip.vision.block_count':27,'clip.vision.projection_dim':2560,
                     'clip.vision.spatial_merge_size':2,'clip.vision.image_size':768,
                     'clip.vision.image_mean':[0.5,0.5,0.5],'clip.vision.image_std':[0.5,0.5,0.5]}
    for key,value in expected_meta.items():
        require(meta.get(key) == value, 'Actual projector metadata differs: ' + key)
    require(meta.get('clip.vision.is_deepstack_layers') == [False] * 27, 'Unexpected active DeepStack weights')
    require(abs(meta['clip.vision.attention.layer_norm_epsilon'] - 1e-6) < 1e-12, 'Unexpected norm epsilon')
    mapping = directory_mapping(CONFIG); validate_directory(data['tensors'], mapping)
    transformer_files = ['__init__.py','vision_utils.py','activations.py','configuration_utils.py',
                         'modeling_utils.py','initialization.py',
                         'models/qwen3_vl/modeling_qwen3_vl.py','models/qwen3_vl/configuration_qwen3_vl.py',
                         'models/qwen2_vl/image_processing_qwen2_vl.py','image_processing_backends.py',
                         'image_processing_utils.py','image_transforms.py','image_utils.py','processing_utils.py']
    source_files = [package_file('transformers',name) for name in transformer_files]
    source_files += [package_file('torch',name) for name in ['__init__.py','version.py','nn/modules/conv.py','nn/modules/normalization.py']]
    source_files += [package_file('gguf',name) for name in ['gguf_reader.py','quants.py','constants.py']]
    source_files += [Path(__file__).resolve(), HERE / 'inspect-remote-gguf.py']
    text = package_file('transformers','__init__.py').read_text(encoding='utf-8')
    code_version = re.search(r'^__version__ = "([^"]+)"',text,re.MULTILINE).group(1)
    require('class Qwen3VLVisionModel' in package_file('transformers','models/qwen3_vl/modeling_qwen3_vl.py').read_text(),
            'Explicit installed reference class unavailable')
    versions = {name: importlib.metadata.version(name) for name in ['torch','transformers','gguf','Pillow','numpy','torchvision']}
    plan = {'Schema':'qwen-hf-vision-oracle-plan.v1','Status':'prepared-not-executed',
            'ProjectorMetadataOnly':metadata(model),'ProjectorHeaderSha256':header_sha,
            'Image':pin(image),'Sources':[pin(path) for path in source_files],
            'DistributionVersions':versions,'TransformersCodeVersion':code_version,
            'TransformersVersionMismatch':versions['transformers'] != code_version,
            'ReferenceClass':'transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionModel',
            'ReferenceConfigClass':'transformers.models.qwen3_vl.configuration_qwen3_vl.Qwen3VLVisionConfig',
            'Config':CONFIG,'Mapping':mapping,'ActualTensorDirectory':data['tensors'],
            'Threads':8,'Dtype':'float32','Attention':'eager','GridTHW':[[1,30,40]],
            'PixelPolicy':{'ImageClass':'PIL RGB uint8 from exact opaque PNG','Resize':False,
                           'ProcessorClass':'transformers.models.qwen2_vl.image_processing_qwen2_vl.Qwen2VLImageProcessor',
                           'PatchSize':16,'TemporalPatchSize':2,'MergeSize':2,'Mean':[0.5]*3,'Std':[0.5]*3},
            'OutputShape':[300,2560],'TimeoutSeconds':600,
            'PrimaryReview':['https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py',
                             'https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen4_exp/configuration_qwen4_exp.py'],
            'Limitations':['Architecture-equivalent installed Qwen3-VL vision-component reference; Qwen4-Exp class is absent locally.',
                           'Not official full Qwen4 model ground truth, answer-quality qualification, performance benchmark, or CUDA validation.',
                           'All334source GGUF tensors must map into333HF parameters; only the two temporal convolution slices combine into one destination.',
                           'Actual source hashes are authoritative; installed Transformers distribution/code versions disagree.',
                           'CPU F32 eager execution only. BF16/CUDA execution is intentionally not implemented or claimed.',
                           'No pass tolerance is selected from resulting model differences; report numerical observations only.']}
    save(out / 'plan.json',plan)


def verify(plan):
    require(metadata(plan['ProjectorMetadataOnly']['Path']) == plan['ProjectorMetadataOnly'], 'Projector metadata changed')
    data, digest = header(plan['ProjectorMetadataOnly']['Path'])
    require(digest == plan['ProjectorHeaderSha256'], 'Projector bounded header changed')
    validate_directory(data['tensors'],plan['Mapping'])
    require(pin(plan['Image']['Path']) == plan['Image'], 'Image bytes changed')
    for source in plan['Sources']:
        require(pin(source['Path']) == source, 'Reference source changed: ' + source['Path'])


def worker(args):
    plan = json.loads(Path(args.plan).read_text(encoding='utf-8')); out = fresh(args.output)
    result = {'Schema':'qwen-hf-vision-oracle-capture.v1','Completed':False,'QualityPassed':None,
              'PerformanceQualified':False,'LanguageModelExecuted':False,'CudaExecuted':False,
              'ArchitectureEquivalentComponentOnly':True,'ProjectorSha256Scope':'Path/size/mtime plus bounded header; not a fresh full907MB hash.'}
    try:
        require(plan['Schema'] == 'qwen-hf-vision-oracle-plan.v1' and plan['Config'] == CONFIG
                and plan['Dtype'] == 'float32' and plan['Attention'] == 'eager', 'Unsupported/changed reference plan')
        verify(plan)
        # Offline imports and CPU-only behavior are explicit before torch loads.
        os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['OMP_NUM_THREADS'] = str(plan['Threads']); os.environ['MKL_NUM_THREADS'] = str(plan['Threads'])
        import numpy as np
        import torch
        import transformers
        from PIL import Image
        from gguf import GGUFReader
        from gguf.quants import dequantize
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        require(transformers.__version__ == plan['TransformersCodeVersion'], 'Imported Transformers code version changed')
        torch.set_num_threads(plan['Threads']); torch.set_num_interop_threads(1)
        torch.set_default_dtype(torch.float32)
        torch.use_deterministic_algorithms(True)
        config = Qwen3VLVisionConfig(**plan['Config']); config._attn_implementation = 'eager'
        started = time.monotonic(); model = Qwen3VLVisionModel(config).eval().to(device='cpu',dtype=torch.float32)
        require(model.__class__.__module__ + '.' + model.__class__.__name__ == plan['ReferenceClass'], 'Wrong imported reference class')
        require(model.merger.use_postshuffle_norm is False and model.merger.act_fn.approximate == 'none'
                and len(model.deepstack_merger_list) == 0 and model.num_grid_per_side == 48
                and model.rotary_pos_emb.theta == 10000.0
                and all(block.attn.is_causal is False and block.attn.num_heads == 16 and block.attn.head_dim == 72
                        for block in model.blocks),
                'Reference graph differs: pre-shuffle norm, erf merger, no DeepStack,48x48 learned positions required')
        targets = model.state_dict()
        require(set(targets) == {row['Destination'] for row in plan['Mapping']}, 'Missing/unexpected HF state parameter')
        reader = GGUFReader(plan['ProjectorMetadataOnly']['Path'],'r')
        sources = {tensor.name:tensor for tensor in reader.tensors}
        loaded = []
        with torch.no_grad():
            for row in plan['Mapping']:
                source = sources[row['Source']]
                values = dequantize(source.data,source.tensor_type).reshape(row['SourceShape'])
                require(values.dtype == np.float32 and np.isfinite(values).all(), 'Invalid actual checkpoint tensor: ' + row['Source'])
                target = targets[row['Destination']]
                if 'TemporalSlice' in row:
                    require(list(target.shape) == [1152,3,2,16,16], 'HF temporal convolution shape mismatch')
                    target = target[:,:,row['TemporalSlice'],:,:]
                require(list(target.shape) == row['SourceShape'], 'HF destination shape mismatch: ' + row['Destination'])
                target.copy_(torch.from_numpy(np.array(values,copy=True)))
                loaded.append(row['Source'])
        require(set(loaded) == set(sources) and len(loaded) == len(sources) == 334, 'Not every GGUF tensor loaded exactly once')
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            require(parameter.device.type == 'cpu' and parameter.dtype == torch.float32, 'Non-CPU/F32 parameter')
        require(all(module.eps == 1e-6 for module in model.modules() if isinstance(module,torch.nn.LayerNorm)), 'Unexpected norm epsilon')
        result['LoadedSourceTensors'] = len(loaded); result['TargetParameters'] = len(targets)
        result['LoadMs'] = (time.monotonic()-started)*1000; result['ActualConfig'] = config.to_dict()
        processor = Qwen2VLImageProcessor(do_resize=False,do_rescale=True,rescale_factor=1/255,
            do_normalize=True,image_mean=[0.5]*3,image_std=[0.5]*3,patch_size=16,temporal_patch_size=2,merge_size=2)
        with Image.open(plan['Image']['Path']) as original:
            require(original.size == (640,480) and original.convert('RGBA').getchannel('A').getextrema() == (255,255),
                    'Original opaque no-resize image changed')
            image = original.convert('RGB')
            inputs = processor(images=image,return_tensors='pt')
        pixels,grid = inputs['pixel_values'],inputs['image_grid_thw']
        require(list(pixels.shape) == [1200,1536] and pixels.dtype == torch.float32 and grid.tolist() == plan['GridTHW'],
                'Independent HF pixel/grid geometry mismatch')
        result['PixelValuesShape'] = list(pixels.shape); result['PixelValuesSha256'] = hashlib.sha256(pixels.contiguous().numpy().tobytes()).hexdigest()
        result['GridTHW'] = grid.tolist()
        stages = []
        def export(name,tensor):
            require(isinstance(tensor,torch.Tensor) and tensor.device.type == 'cpu' and tensor.dtype == torch.float32
                    and torch.isfinite(tensor).all().item(), 'Nonfinite/non-F32 stage: ' + name)
            path = out / (name + '.f32'); tensor.detach().contiguous().numpy().tofile(path)
            stages.append({'Name':name,'Shape':list(tensor.shape),'File':pin(path)})
        hooks = [model.patch_embed.register_forward_hook(lambda module,inputs,value:export('patch-embed',value)),
                 model.blocks[0].register_forward_pre_hook(lambda module,inputs:export('after-position-add',inputs[0])),
                 model.blocks[0].register_forward_hook(lambda module,inputs,value:export('block-0',value)),
                 model.blocks[26].register_forward_hook(lambda module,inputs,value:export('block-26',value))]
        with torch.inference_mode():
            started = time.monotonic(); output = model(pixels,grid_thw=grid,return_dict=True)
            result['ForwardDiagnosticMs'] = (time.monotonic()-started)*1000
            projection = output.pooler_output
            require(list(projection.shape) == plan['OutputShape'], 'HF projection shape mismatch')
            export('projected',projection)
            # These are vision patch rotary coordinates, not LLM decoder THW.
            from transformers.vision_utils import get_vision_position_ids
            positions = get_vision_position_ids(grid,2)
            save(out / 'vision-positions-hw.json',positions.tolist())
        for hook in hooks:
            hook.remove()
        result['Stages'] = stages; result['AllFinite'] = True
        result['TorchRuntimeVersion'] = torch.__version__; result['TransformersRuntimeVersion'] = transformers.__version__
        verify(plan); result['SourceImageProjectorBookendsUnchanged'] = True; result['Completed'] = True
    except Exception as error:
        import traceback
        result['Error'] = str(error); result['Traceback'] = traceback.format_exc()
    save(out / 'capture.json',result)
    return 0 if result['Completed'] else 1


def run(args):
    plan_path = Path(args.plan).resolve(); plan = json.loads(plan_path.read_text(encoding='utf-8')); out = fresh(args.output)
    report = {'Schema':'qwen-hf-vision-oracle-run.v1','Completed':False,'QualityPassed':None,'PerformanceQualified':False,
              'Plan':pin(plan_path)}
    try:
        verify(plan)
        require(plan['TimeoutSeconds'] == 600, 'Unexpected execution bound')
        command = [sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),'worker',
                   '--plan',str(plan_path),'--output',str(out / 'capture')]
        report['Command'] = command
        with (out / 'stdout.log').open('wb') as stdout,(out / 'stderr.log').open('wb') as stderr:
            with subprocess.Popen(command,cwd=out,stdout=stdout,stderr=stderr) as child:
                report['OwnedPid'] = child.pid
                try:
                    report['ExitCode'] = child.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait(); report['TimedOut'] = True; raise
                finally:
                    report['OwnedChildExited'] = child.poll() is not None
        capture = json.loads((out / 'capture/capture.json').read_text(encoding='utf-8'))
        report['Capture'] = capture
        require(report['ExitCode'] == 0 and capture['Completed'], 'Independent HF component capture did not complete')
        verify(plan); report['Completed'] = True
    except Exception as error:
        report['Error'] = str(error)
    save(out / 'report.json',report)
    return 0 if report['Completed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__); modes = parser.add_subparsers(dest='mode',required=True)
    p = modes.add_parser('prepare'); p.add_argument('--mmproj',type=Path,required=True); p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.set_defaults(action=prepare)
    for name,action in [('run',run),('worker',worker)]:
        p = modes.add_parser(name); p.add_argument('--plan',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.set_defaults(action=action)
    args = parser.parse_args(); return args.action(args)


if __name__ == '__main__':
    sys.exit(main())
