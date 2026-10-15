#!/usr/bin/env python3
"""Rebuild owned translation units selected by recorded Ninja dependencies.

Windows/MSVC validation utility. It never configures/compiles ggml or overwrites
the completed base build. Fresh DLLs are not model/device/performance acceptance.
Prepare pins dependency/source/object closure; run consumes it in a VC environment.
"""
import argparse
import copy
import ctypes
import difflib
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import subprocess
import sys
import time
import traceback


CACHE = 'TensorSharp.GGML.Native/ggml_ops_host_moe_cache.cpp'
CMAKE = 'TensorSharp.GGML.Native/CMakeLists.txt'
MODES = ('production','hooks')
LIBRARIES = ('ggml.lib','ggml-base.lib','ggml-cpu.lib','ggml-cuda.lib')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):digest.update(chunk)
    return digest.hexdigest()


def pin(path):
    path=Path(path).resolve(strict=True)
    return dict(Path=str(path),Bytes=path.stat().st_size,Sha256=sha(path))


def read(path):
    path=Path(path)
    if path.stat().st_size>8<<20:raise ValueError('Expected bounded build evidence JSON.')
    return json.loads(path.read_text('utf-8-sig'))


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n','utf-8')


def utc():return datetime.now(timezone.utc).isoformat()


def canonical(value):return str(value).replace('\\','/').rstrip('/').casefold()


def absolute(value,directory):
    path=Path(value)
    return path.resolve() if path.is_absolute() else (Path(directory)/path).resolve()


def windows_argv(command):
    if os.name!='nt':raise RuntimeError('Windows CommandLineToArgvW is required; never approximate shell parsing.')
    shell=ctypes.WinDLL('shell32',use_last_error=True)
    shell.CommandLineToArgvW.argtypes=[ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_int)]
    shell.CommandLineToArgvW.restype=ctypes.POINTER(ctypes.c_wchar_p)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.LocalFree.argtypes=[ctypes.c_void_p];kernel.LocalFree.restype=ctypes.c_void_p
    count=ctypes.c_int();argv=shell.CommandLineToArgvW(command,ctypes.byref(count))
    if not argv:raise ctypes.WinError(ctypes.get_last_error())
    try:return [argv[index] for index in range(count.value)]
    finally:kernel.LocalFree(ctypes.cast(argv,ctypes.c_void_p))


def strip_link_shell(command):
    match=re.fullmatch(r'[^\r\n]*cmd\.exe /C "cd \. && (.+) && cd \."',command,re.I)
    if not match:raise ValueError('Require the exact known Ninja vs_link_dll wrapper.')
    inner=match.group(1)
    if re.search(r'(?:^|\s)(?:&&|\|\||[<>|])(?:\s|$)',inner):
        raise ValueError('Unexpected nested shell operator in native link command.')
    return inner


def rewrite_owned_compile(row,base_build,new_build,base_driver,new_driver):
    argv=windows_argv(row['command'])
    compiler=Path(argv[0]).name.lower()
    if compiler not in ('cl.exe','nvcc.exe'):raise ValueError('Require the pinned CL or NVCC compiler.')
    source=Path(row['file']).resolve()
    if not source.is_relative_to(Path(base_driver).resolve()) or source.suffix not in ('.cpp','.cu','.c'):
        raise ValueError('Only TensorSharp-owned translation units may be compiled.')
    suffix=source.relative_to(Path(base_driver).resolve())
    expected=absolute(row['output'],base_build)
    if not expected.is_relative_to(Path(base_build).resolve()) or '/ggmlops.dir/' not in canonical(expected):
        raise ValueError('Owned object is outside the pinned GgmlOps target.')
    destination=Path(new_build)/expected.relative_to(Path(base_build).resolve())
    found={'source':0,'output':0,'fd':0,'mt':0,'mf':0};changed=[];index=0
    while index<len(argv):
        value=argv[index];low=value.lower()
        if canonical(value)==canonical(source):value=str(Path(new_driver)/suffix);found['source']+=1
        elif low.startswith('/fo'):
            if canonical(absolute(value[3:],base_build))!=canonical(expected):raise ValueError('CL output differs from its compilation database.')
            value='/Fo'+str(destination);found['output']+=1
        elif low.startswith('/fd'):
            old=absolute(value[3:],base_build)
            if not old.is_relative_to(Path(base_build).resolve()):raise ValueError('CL PDB output is outside the pinned base build.')
            value='/Fd'+str(Path(new_build)/'CMakeFiles/GgmlOps.dir')+'\\';found['fd']+=1
        elif compiler=='nvcc.exe' and value in ('-o','-MT','-MF'):
            if index+1>=len(argv):raise ValueError('Incomplete NVCC output option.')
            old=absolute(argv[index+1],base_build)
            wanted=Path(str(expected)+'.d') if value=='-MF' else expected
            if canonical(old)!=canonical(wanted):raise ValueError('Unexpected NVCC object/dependency output.')
            changed.extend([value,str(Path(str(destination)+'.d') if value=='-MF' else destination)])
            found[{'-o':'output','-MT':'mt','-MF':'mf'}[value]]+=1;index+=2;continue
        elif compiler=='nvcc.exe' and low.startswith('-xcompiler=-fd'):
            value='-Xcompiler=-Fd'+str(Path(new_build)/'CMakeFiles/GgmlOps.dir')+'\\,-FS';found['fd']+=1
        changed.append(value)
        index+=1
    wanted={'source':1,'output':1,'fd':1,'mt':1 if compiler=='nvcc.exe' else 0,'mf':1 if compiler=='nvcc.exe' else 0}
    if found!=wanted:raise ValueError('Require exact source, object, compiler-PDB and NVCC dependency outputs.')
    if any(canonical(value).startswith(canonical(base_build)) for value in changed):
        raise ValueError('CL may not write into the immutable base build.')
    return changed,destination


def rewrite_compile(row,base_build,new_build,base_driver,new_driver):
    if Path(row['file']).name!=Path(CACHE).name:raise ValueError('Not the exact cache translation unit.')
    return rewrite_owned_compile(row,base_build,new_build,base_driver,new_driver)


def rewrite_link(command,base_build,new_build,objects,new_cache,definition,object_paths=None):
    argv=windows_argv(strip_link_shell(command))
    if len(argv)<8 or Path(argv[0]).name.lower()!='cmake.exe' or argv[1:3]!=['-E','vs_link_dll']:
        raise ValueError('Require the exact pinned CMake native DLL link driver.')
    mapping={canonical(Path(base_build)/relative):Path(base_build)/relative for relative in objects}
    cache=[key for key in mapping if key.endswith('/ggml_ops_host_moe_cache.cpp.obj')]
    if len(cache)!=1:raise ValueError('Require exactly one old cache object.')
    if object_paths is None:mapping[cache[0]]=Path(new_cache)
    else:
        if set(object_paths)!=set(objects):raise ValueError('Missing final owned object provenance.')
        mapping={canonical(Path(base_build)/name):Path(value) for name,value in object_paths.items()}
    found_objects=[];counts={'out':0,'implib':0,'pdb':0,'intdir':0,'def':0};result=[]
    for value in argv:
        low=value.lower()
        if low.endswith('.obj'):
            key=canonical(absolute(value,base_build))
            if key not in mapping or key in found_objects:raise ValueError('Missing, duplicate or unknown linked object.')
            found_objects.append(key);value=str(mapping[key])
        elif low.startswith('--intdir='):
            if canonical(absolute(value.split('=',1)[1],base_build))!=canonical(Path(base_build)/'CMakeFiles/GgmlOps.dir'):
                raise ValueError('Unexpected original manifest intermediate directory.')
            value='--intdir='+str(Path(new_build)/'CMakeFiles/GgmlOps.dir');counts['intdir']+=1
        elif low.startswith('/def:'):
            if canonical(absolute(value[5:],base_build))!=canonical(Path(base_build)/'ggml_api_exports.def'):
                raise ValueError('Unexpected original linker export definition.')
            value='/DEF:'+str(definition);counts['def']+=1
        else:
            for flag,name in (('/out:','GgmlOps.dll'),('/implib:','GgmlOps.lib'),('/pdb:','GgmlOps.pdb')):
                if low.startswith(flag):
                    if canonical(absolute(value[len(flag):],base_build))!=canonical(Path(base_build)/name):
                        raise ValueError('Unexpected original native linker output.')
                    value=flag+str(Path(new_build)/name);counts[flag[1:-1]]+=1;break
        if value.startswith('@') or value in ('&&','|','>','<'):
            raise ValueError('Response files and shell interpretation are unsupported.')
        result.append(value)
    if len(found_objects)!=len(objects) or set(found_objects)!=set(mapping) or any(value!=1 for value in counts.values()):
        raise ValueError('Require every exact object once and all fresh native link outputs.')
    return result


def check_object_closure(build,expected):
    if len(expected)!=69:raise ValueError('Require the completed 69-object owned native base.')
    actual={str(path.relative_to(build)).replace('\\','/'):sha(path)
            for path in (Path(build)/'CMakeFiles/GgmlOps.dir').rglob('*.obj')}
    if actual!=expected:raise ValueError('A pinned base object is missing, extra, or changed.')
    return actual


def cache_value(path,name):
    matches=re.findall(r'^'+re.escape(name)+r':[^=]+=([^\r\n]+)$',Path(path).read_text('utf-8-sig'),re.M)
    if len(matches)!=1:raise ValueError('Missing exact base CMake cache field: '+name)
    return matches[0]


def validate_live_sources(current,base,overlays,allow_owned_changes=False,declared_test_configuration=False):
    if set(current)!=set(base) and not allow_owned_changes:
        raise ValueError('Owned source closure changed; configure a fresh complete native build.')
    for relative in overlays:
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts or path.parts[:2]!=('TensorSharp.GGML.Native','tests') \
                or path.suffix not in ('.cpp','.inc','.h') or relative not in base:
            raise ValueError('Only explicitly named existing test fixtures may be overlaid.')
    changes={name for name in current.keys()|base.keys() if current.get(name)!=base.get(name)}
    if not allow_owned_changes and changes-({CACHE}|set(overlays)):
        raise ValueError('A non-cache translation unit/header changed; all affected objects must be rebuilt.')
    configuration={CMAKE} if declared_test_configuration else set()
    if allow_owned_changes and any(Path(name).suffix not in ('.cpp','.cu','.c','.h','.hpp','.inc') for name in changes-set(overlays)-configuration):
        raise ValueError('Build configuration/export/non-code changes require a fresh configured full build.')
    return sorted(changes)


def declared_test_additions(original,current,new_paths):
    """Prove only standalone test target insertions; preserve all original text.

    This deliberately is not a general CMake interpreter. Only simple, single-
    line test commands are allowed inside the original BUILD_TESTS block.
    Production target flags, existing commands and arbitrary configuration are
    never accepted by this exemption.
    """
    before=original.splitlines(keepends=True);after=current.splitlines(keepends=True)
    guards=[];stack=[]
    for line in before:
        stripped=line.strip()
        match=re.match(r'^if\s*\(',stripped,re.I)
        if match:stack.append(bool(re.fullmatch(r'if\s*\(\s*TENSORSHARP_GGML_NATIVE_BUILD_TESTS\s*\)\s*(?:#.*)?',stripped,re.I)))
        elif re.match(r'^(?:else|elseif)\s*\(',stripped,re.I):
            if not stack:raise ValueError('Unproven CMake conditional scope.')
            stack[-1]=False
        elif re.match(r'^endif\s*\(',stripped,re.I):
            if not stack:raise ValueError('Unproven CMake conditional scope.')
            stack.pop()
        guards.append(any(stack))
    if stack:raise ValueError('Unclosed CMake conditional scope.')
    inserted=[]
    for tag,i,j,a,b in difflib.SequenceMatcher(a=before,b=after,autojunk=False).get_opcodes():
        if tag=='equal':continue
        if tag!='insert' or i==0 or not guards[i-1]:
            raise ValueError('Only additive standalone test commands inside the existing BUILD_TESTS guard are allowed.')
        inserted.extend(after[a:b])
    target_source={};test_targets={};includes=[];properties=[]
    for line in inserted:
        value=line.strip()
        if not value:continue
        match=re.fullmatch(r'add_executable\(([A-Za-z][A-Za-z0-9_]*) (tests/[A-Za-z0-9_/-]+\.cpp)\)',value)
        if match:
            target,path=match.groups();relative='TensorSharp.GGML.Native/'+path
            if '..' in Path(path).parts or relative not in new_paths or target in target_source \
                    or re.search(r'\b'+re.escape(target)+r'\b',original):
                raise ValueError('Added test target must uniquely declare a new tests/*.cpp source.')
            target_source[target]=relative;continue
        match=re.fullmatch(r'add_test\(NAME ([A-Za-z0-9_-]+) COMMAND ([A-Za-z][A-Za-z0-9_]*)\)',value)
        if match:
            name,target=match.groups()
            if name in test_targets or re.search(r'\b'+re.escape(name)+r'\b',original):
                raise ValueError('Added test name must be new and unique.')
            test_targets[name]=target;continue
        match=re.fullmatch(r'target_include_directories\(([A-Za-z][A-Za-z0-9_]*) PRIVATE "\$\{CMAKE_CURRENT_SOURCE_DIR\}"\)',value)
        if match:includes.append(match[1]);continue
        match=re.fullmatch(r'set_tests_properties\(([A-Za-z0-9_-]+) PROPERTIES SKIP_RETURN_CODE 77\)',value)
        if match:properties.append(match[1]);continue
        raise ValueError('Unrecognized added CMake command requires a fresh configured native build.')
    if not target_source or set(test_targets.values())!=set(target_source) \
            or len(test_targets)!=len(target_source) or set(includes)-set(target_source) \
            or set(properties)-set(test_targets):
        raise ValueError('Added standalone test targets/tests are incomplete or affect unknown targets.')
    return dict(Valid=True,SourcePaths=sorted(target_source.values()),Targets=target_source,Tests=test_targets,
                Scope='Insertion-only standalone tests under the original BUILD_TESTS guard; all existing CMake lines/production target configuration remain unchanged after newline normalization.')


def regenerate_driver_cmake(formal,replacements,original_formal,original_driver):
    if not replacements:raise ValueError('Missing configured ancestor driver transformation proof.')
    def transform(value):
        for old,new in replacements.items():
            if old not in value:raise ValueError('An inherited driver transformation anchor is missing.')
            value=value.replace(old,new)
        return value
    if transform(original_formal)!=original_driver:
        raise ValueError('Recorded driver transformations do not reproduce the immutable base CMake input.')
    return transform(formal)


def parse_ninja_dependencies(text):
    """Consume recorded deps exactly; no inference from #include source text."""
    result={};current=None
    def finish():
        if current is None:return
        key=canonical(current['Target'])
        if key in result:raise ValueError('Duplicate Ninja dependency target.')
        if len(current['Dependencies'])!=current['Count']:
            raise ValueError('Incomplete Ninja dependency list.')
        result[key]=current.copy()
    for line in text.splitlines():
        if not line.strip():continue
        match=re.fullmatch(r'(.+): #deps ([0-9]+), deps mtime ([0-9]+) \((VALID|STALE)\)',line)
        if match:
            finish();current=dict(Target=match[1],Count=int(match[2]),Mtime=int(match[3]),
                                  Valid=match[4]=='VALID',Dependencies=[])
        elif line.startswith('    ') and current is not None:
            current['Dependencies'].append(line[4:])
        else:raise ValueError('Unknown Ninja dependency output; full recompilation required.')
    finish();return result


def dependency_path(value):
    """Only literal absolute compiler paths; never expand make/shell macros."""
    if (not value or any(char in value for char in '\r\n\0$"\'|<>*?')
            or value.endswith(('\\','/'))):
        raise ValueError('Invalid literal compiler dependency path.')
    windows=PureWindowsPath(value)
    if windows.is_absolute():
        if ':' in value[2:]:raise ValueError('Unexpected dependency colon.')
        return str(windows)
    if value.startswith('/') and ':' not in value:return value
    raise ValueError('Compiler dependencies must use absolute paths.')


def parse_cl_dependencies(text, source):
    if len(text)>8<<20 or '\0' in text:raise ValueError('Expected bounded complete CL include output.')
    paths=[dependency_path(source)];includes=0
    for line in text.splitlines():
        if line.startswith('Note: including file:'):
            value=line[len('Note: including file:'):].strip()
            paths.append(dependency_path(value));includes+=1
        elif 'including file:' in line:
            raise ValueError('Unrecognized CL include marker.')
    if not includes:raise ValueError('Missing authoritative CL include markers.')
    return list(dict.fromkeys(paths))


def parse_nvcc_dependencies(text, expected_target, expected_source):
    """One literal Make rule, including NVCC's raw Windows target separators."""
    if len(text)>8<<20 or any(char in text for char in '\0$"\'|<>*?'):
        raise ValueError('Expected bounded literal NVCC dependency rule.')
    logical=re.sub(r'\\\r?\n', ' ', text).strip()
    if '\n' in logical or '\r' in logical or not logical:
        raise ValueError('Require exactly one complete NVCC dependency rule.')
    def tokens(value, allow_rule):
        output=[];word=[];delimiter=None;i=0
        while i<len(value):
            char=value[i]
            if char in ' \t':
                if word:output.append(''.join(word));word=[]
                i+=1;continue
            if char=='\\':
                if i+1>=len(value):raise ValueError('Incomplete NVCC dependency escape.')
                following=value[i+1]
                if following in ' \t:#':word.append(following);i+=2;continue
                # NVCC emits raw E:\\directory separators in its target. They
                # are literal Windows path separators, not unknown make escapes.
                if not (len(word)>=2 and word[0].isalpha() and word[1]==':'
                        or (not word and following=='\\') or (word and word[0]=='\\')):
                    raise ValueError('Unknown NVCC dependency escape.')
                word.append(char);i+=1;continue
            if char=='#':raise ValueError('Unexpected unescaped Make comment.')
            if char==':':
                drive=len(word)==1 and word[0].isalpha() and i+1<len(value) and value[i+1] in '/\\'
                if not drive:
                    if not allow_rule or delimiter is not None:raise ValueError('Multiple/unknown NVCC dependency targets.')
                    if word:output.append(''.join(word));word=[]
                    delimiter=len(output);i+=1;continue
            word.append(char);i+=1
        if word:output.append(''.join(word))
        return output,delimiter
    words,split=tokens(logical,True)
    if split!=1 or len(words)<2:raise ValueError('Require one target and a nonempty dependency list.')
    target=dependency_path(words[0]);wanted=dependency_path(str(expected_target))
    if canonical(target)!=canonical(wanted):raise ValueError('NVCC dependency target differs from compiled object.')
    paths=[dependency_path(word) for word in words[1:]]
    source=dependency_path(str(expected_source))
    if canonical(source) not in {canonical(path) for path in paths}:
        raise ValueError('NVCC dependency list omits its exact compiled source.')
    return list(dict.fromkeys(paths))


def recorded_compiler_dependencies(origin,mode,dependency_files):
    """Bind compiler include evidence to the successful exact object generation."""
    summary=origin['Summary'];root=origin['Root'];link=summary.get('IncrementalPlan')
    if (summary.get('passed') is not True or summary.get('no_upstream_compilation') is not True
            or summary.get('before')!=summary.get('after') or not link):return None
    if pin(link['Path'])!=link:raise ValueError('Compiler dependency origin plan changed.')
    plan=read(link['Path']);entry=plan['Modes'][mode];build=root/('build-'+mode)
    if (canonical(plan['Output'])!=canonical(root) or plan['source_snapshot']!=summary['source_snapshot']
            or plan['before']!=summary['before']):
        raise ValueError('Compiler dependency origin plan/source/bookends differ from its completed receipt.')
    if canonical(entry['NewBuild'])!=canonical(build):raise ValueError('Compiler dependency origin build differs.')
    receipt=pin(root/'build-summary.json');dependency_files[receipt['Path']]=receipt
    dependency_files[link['Path']]=link
    records={}
    for index,row in enumerate(entry.get('CompileCommands',[]),1):
        target=row['Target'];key=canonical(target);source=root/'driver'/Path(row['Source']).relative_to('TensorSharp.GGML.Native')
        output=absolute(row['Output'],build);argv=row['Command']
        if key in records or canonical(output)!=canonical(build/target) or canonical(source) not in [canonical(arg) for arg in argv]:
            raise ValueError('Compiler dependency target/source/output identity differs.')
        if sum(canonical(arg)==canonical(source) for arg in argv)!=1:raise ValueError('Ambiguous compiler source argument.')
        if row['Source'] not in summary['source_snapshot'] or sha(source)!=summary['source_snapshot'][row['Source']]:
            raise ValueError('Exact compiled source generation changed.')
        step_name='compile-'+mode+'-'+str(index)
        steps=[step for step in summary['steps'] if step['name']==step_name]
        if (len(steps)!=1 or steps[0]['exit_code']!=0 or steps[0]['command']!=argv
                or canonical(steps[0]['cwd'])!=canonical(build)):raise ValueError('Exact successful dependency compiler step unavailable.')
        log=root/(step_name+'.log');log_pin=pin(log)
        if log_pin['Bytes']>8<<20 or log_pin['Sha256']!=steps[0]['log_sha256']:raise ValueError('Compiler include log changed or exceeds bound.')
        object_pin=pin(output)
        if object_pin['Sha256']!=summary[mode+'_objects'].get(target):raise ValueError('Compiler dependency object changed.')
        provenance=summary.get(mode+'_object_provenance',{}).get(target)
        if (not provenance or provenance.get('Recompiled') is not True
                or canonical(provenance['Path'])!=canonical(output) or provenance['Sha256']!=object_pin['Sha256']):
            raise ValueError('Compiler dependencies do not belong to this successful object generation.')
        compiler=Path(argv[0]).name.lower();evidence={'BuildReceipt':receipt,'IncrementalPlan':link,'CompileLog':log_pin,
            'Object':object_pin,'ExactSuccessfulCommand':argv,'Target':target,'Source':str(source)}
        if compiler=='cl.exe':
            if argv.count('/showIncludes')!=1:raise ValueError('Require actual CL dependency capture flag.')
            outputs=[absolute(arg[3:],build) for arg in argv if arg.lower().startswith('/fo')]
            if len(outputs)!=1 or canonical(outputs[0])!=canonical(output):raise ValueError('CL captured output differs from compiled target.')
            paths=parse_cl_dependencies(log.read_text('utf-8-sig'),str(source))
        elif compiler=='nvcc.exe':
            if argv.count('-MD')!=1 or argv.count('-MT')!=1 or argv.count('-MF')!=1:
                raise ValueError('Require actual complete NVCC dependency capture flags.')
            mt=absolute(argv[argv.index('-MT')+1],build);dep=absolute(argv[argv.index('-MF')+1],build)
            if argv.count('-o')!=1 or canonical(absolute(argv[argv.index('-o')+1],build))!=canonical(output):
                raise ValueError('NVCC captured output differs from compiled target.')
            if canonical(mt)!=canonical(output) or canonical(dep)!=canonical(str(output)+'.d'):
                raise ValueError('NVCC dependency command differs from compiled target.')
            dep_pin=pin(dep)
            if dep_pin['Bytes']>8<<20:raise ValueError('NVCC dependency evidence exceeds bound.')
            if dep.stat().st_mtime_ns>output.stat().st_mtime_ns:
                raise ValueError('NVCC dependency evidence is newer than its compiled object generation.')
            paths=parse_nvcc_dependencies(dep.read_text('utf-8-sig'),output,source)
            evidence['NvccDependencies']=dep_pin;dependency_files[dep_pin['Path']]=dep_pin
        else:raise ValueError('Unsupported authoritative dependency compiler.')
        dependency_files[log_pin['Path']]=log_pin
        dependency_files[object_pin['Path']]=object_pin
        records[key]=dict(Target=target,Count=len(paths),Valid=True,Dependencies=paths,CompilerEvidence=evidence)
    return records or None


def rebuild_decisions(units,changes,force_all=False,force_targets=()):
    """Any unproven proposed reuse conservatively rebuilds the complete mode."""
    forced=set(force_targets)
    if len(forced)!=len(force_targets) or forced-set(units):
        raise ValueError('Require unique exact recorded owned targets to recompile.')
    decisions={};missing=[]
    for name,unit in units.items():
        if name in forced:
            decisions[name]=dict(Rebuild=True,Reason='explicit-owned-target-recompile',Affected=[]);continue
        source=unit['Source']
        if source in changes:
            decisions[name]=dict(Rebuild=True,Reason='owned-translation-unit-changed',Affected=[source]);continue
        proof=unit.get('DependencyProof',{})
        if proof.get('Valid') is not True:
            missing.append(name);continue
        affected=sorted(set(proof['OwnedDependencies'])&set(changes))
        decisions[name]=dict(Rebuild=bool(affected),Reason='recorded-owned-header-changed' if affected else 'recorded-dependency-closure-unchanged',Affected=affected)
    if missing or force_all:
        reason='explicit-full-owned-recompile' if force_all else 'dependency-proof-insufficient-full-owned-recompile'
        return {name:dict(Rebuild=True,Reason=reason,UnprovenReuseTargets=sorted(missing),
                          OriginalReason=decisions.get(name,{}).get('Reason')) for name in units}
    return decisions


def build_chain(root):
    nodes=[];seen=set()
    while True:
        root=Path(root).resolve(strict=True)
        if canonical(root) in seen or len(nodes)>=8:raise ValueError('Invalid recursive base build provenance.')
        seen.add(canonical(root));receipt=root/'build-summary.json';summary=read(receipt)
        if summary.get('passed') is not True or summary.get('no_upstream_compilation') is not True or summary['before']!=summary['after']:
            raise ValueError('Every build ancestor must be completed, unchanged and upstream-unmodified.')
        nodes.append(dict(Root=root,Summary=summary,Receipt=pin(receipt)))
        if not summary.get('IncrementalPlan'):break
        link=summary['IncrementalPlan']
        if pin(link['Path'])!=link:raise ValueError('Base incremental plan changed.')
        plan=read(link['Path']);origin=plan['BaseReceipt']
        if pin(origin['Path'])!=origin:raise ValueError('Ancestor base receipt changed.')
        nodes[-1]['Plan']=plan;nodes[-1]['PlanPin']=link
        root=Path(origin['Path']).parent
    return nodes


def object_provenance(nodes,mode):
    base=nodes[0];expected=base['Summary'][mode+'_objects']
    if len(expected)!=69:raise ValueError('Require the complete original 69 owned objects.')
    supplied=base['Summary'].get(mode+'_object_provenance',{})
    result={}
    for name,digest in expected.items():
        path=Path(supplied[name]['Path']) if supplied else base['Root']/('build-'+mode)/name
        origins=[node for node in nodes if canonical(path)==canonical(node['Root']/('build-'+mode)/name)]
        if len(origins)!=1 or not path.is_file() or sha(path)!=digest:
            raise ValueError('Missing/changed/unqualified base object provenance: '+name)
        if supplied and supplied[name]['Sha256']!=digest:raise ValueError('Base object identity fields disagree.')
        result[name]=dict(Path=str(path),Sha256=digest,Origin=origins[0])
    return result


def qualified_ninja_command(row,commands,build):
    expected=windows_argv(row['command']);source=canonical(row['file'])
    candidates=[]
    for command in commands:
        if 'vs_link_dll' in command:continue
        actual=windows_argv(command)
        if sum(canonical(value)==source for value in actual)==1:candidates.append((command,actual))
    if len(candidates)!=1:raise ValueError('Require one actual Ninja compile command per owned translation unit.')
    command,actual=candidates[0];comparison=list(actual)
    if Path(actual[0]).name.lower()=='cl.exe':
        if comparison.count('/showIncludes')!=1:raise ValueError('Missing exact Ninja CL dependency capture flag.')
        comparison.remove('/showIncludes')
    elif Path(actual[0]).name.lower()=='nvcc.exe':
        if comparison.count('-MD')!=1:raise ValueError('Missing exact Ninja NVCC dependency capture flags.')
        index=comparison.index('-MD')
        if comparison[index:index+2]!=['-MD','-MT'] or len(comparison)<=index+4 or comparison[index+3]!='-MF':
            raise ValueError('Unexpected Ninja NVCC dependency capture flags.')
        expected_output=absolute(row['output'],build)
        if canonical(absolute(comparison[index+2],build))!=canonical(expected_output) \
                or canonical(absolute(comparison[index+4],build))!=canonical(str(expected_output)+'.d'):
            raise ValueError('Ninja NVCC dependency outputs differ from the frozen compilation database.')
        del comparison[index:index+5]
    else:raise ValueError('Unknown native compiler in actual Ninja command.')
    if comparison!=expected:raise ValueError('Actual Ninja compiler flags differ from the frozen compilation database.')
    return command


def command_rows(nodes,mode):
    # Full configured ancestor commands remain authoritative for target/flags.
    ancestor=next((node for node in nodes if (node['Root']/('build-'+mode)/'compile_commands.json').is_file()),None)
    if ancestor is None:raise ValueError('No frozen configured compile database for full fallback.')
    build=ancestor['Root']/('build-'+mode);database=build/'compile_commands.json'
    rows={};commands=ancestor['Root']/(mode+'-actual-ninja-commands.log')
    lines=commands.read_text('utf-8-sig').splitlines()
    for row in read(database):
        output=absolute(row['output'],build)
        if not output.is_relative_to(build) or '/ggmlops.dir/' not in canonical(output):continue
        name=output.relative_to(build).as_posix()
        source=Path(row['file']).resolve()
        if not source.is_relative_to(ancestor['Root']/'driver'):
            raise ValueError('Compile database target contains non-owned/upstream source.')
        if name in rows:raise ValueError('Duplicate frozen owned compile target.')
        effective=dict(row);effective['command']=qualified_ninja_command(row,lines,build)
        rows[name]=effective
    if len(rows)!=69:raise ValueError('Frozen compilation database lacks complete owned target closure.')
    return ancestor,rows,pin(database),pin(commands)


def recorded_dependencies(origin,mode,ninja,dependency_files):
    build=origin['Root']/('build-'+mode);deps=build/'.ninja_deps'
    records={}
    if deps.is_file() and (build/'build.ninja').is_file():
        before=pin(deps)
        text=subprocess.check_output([ninja,'-C',str(build),'-t','deps'],text=True,timeout=60)
        if pin(deps)!=before:raise ValueError('Ninja dependency metadata changed during read-only capture.')
        dependency_files[str(deps)]=before
        try:records=parse_ninja_dependencies(text)
        except ValueError:pass
    # Direct owned compilations retain real compiler dependency evidence even
    # without a configured Ninja tree. Unknown evidence never guesses includes.
    if not records or any(not record['Valid'] for record in records.values()):
        try:compiler=recorded_compiler_dependencies(origin,mode,dependency_files) or {}
        except (OSError,ValueError,KeyError,TypeError,IndexError,UnicodeError):compiler={}
        for target,record in compiler.items():
            if target not in records or not records[target]['Valid']:records[target]=record
    return records or None


def prove_owned_dependencies(record,origin,mode,base_source,source_name,repository):
    if not record or not record['Valid']:return dict(Valid=False,Reason='missing-or-stale-recorded-ninja-deps')
    source=origin['Summary']['source_snapshot']
    if source.get(source_name)!=base_source.get(source_name):return dict(Valid=False,Reason='object-source-generation-not-proven')
    build=origin['Root']/('build-'+mode);driver=origin['Root']/'driver';repo=Path(repository)
    owned=[]
    for value in record['Dependencies']:
        path=absolute(value,build)
        if path.is_relative_to(driver):
            relative='TensorSharp.GGML.Native/'+path.relative_to(driver).as_posix()
            if relative not in source:return dict(Valid=False,Reason='recorded-owned-header-not-in-frozen-source-closure')
            expected=origin['Summary']['generated_driver_files'].get('driver/'+path.relative_to(driver).as_posix(),source[relative])
            if not path.is_file() or sha(path)!=expected:return dict(Valid=False,Reason='recorded-frozen-driver-header-changed')
        elif path.is_relative_to(repo/'TensorSharp.GGML.Native'):
            relative=path.relative_to(repo).as_posix()
            if relative not in source:return dict(Valid=False,Reason='recorded-live-owned-header-not-pinned')
        else:continue
        if source.get(relative)!=base_source.get(relative):
            return dict(Valid=False,Reason='recorded-header-generation-not-valid-for-base-object')
        owned.append(relative)
    result=dict(Valid=True,OwnedDependencies=sorted(set(owned)),RecordedDependencyCount=record['Count'],
                Scope='Recorded VALID Ninja dependency closure for owned source/header changes; no textual include guessing. System/vendor headers are not a historical full physical or toolchain ownership proof.')
    if record.get('CompilerEvidence'):
        result['CompilerEvidence']=record['CompilerEvidence']
        result['Scope']='Exact successful object/compiler command bound to CL /showIncludes or NVCC .d paths; frozen owned source/header generation checked. No textual include guessing or full vendor/toolchain physical ownership proof.'
    return result


def live_pins(repository,base,libraries):
    repo=Path(repository);up=repo/'ExternalProjects/ggml'
    git=lambda *args:subprocess.check_output(['git','-C',str(up),*args],text=True).strip()
    names=sorted(value for value in subprocess.check_output(['git','-C',str(up),'ls-files','-z']).decode().split('\0') if value)
    expected=base['after']['upstream']
    dependency=dict(revision=git('rev-parse','HEAD'),status=git('status','--porcelain'),files={name:sha(up/name) for name in names})
    if len(names)!=2321 or dependency!=expected:
        raise ValueError('Upstream revision/status/2321 tracked file hashes differ from the unchanged completed base.')
    imported={name:sha(Path(libraries)/name) for name in LIBRARIES}
    if imported!=base['after']['libraries']:raise ValueError('An imported upstream library changed.')
    owned=subprocess.check_output(['git','-C',str(repo),'ls-files','--cached','--others','--exclude-standard','-z',
                                   '--','TensorSharp.GGML.Native']).decode().split('\0')
    sources={name:sha(repo/name) for name in sorted(set(owned)) if name and (repo/name).is_file()}
    return dict(upstream=dependency,libraries=imported,current_sources=sources,
                reference_compile_commands=copy.deepcopy(base['after']['reference_compile_commands']))


def declared_added_exports(declarations):
    """Explicit individual owned symbols, never export-name patterns."""
    result = {mode: [] for mode in MODES}
    if not isinstance(declarations, (list, tuple)) or len(declarations) > 32:
        raise ValueError('Require at most32explicit added-export declarations.')
    for declaration in declarations:
        if not isinstance(declaration, str) or declaration.count(':') != 1:
            raise ValueError('An added export must be an exact mode:TSGgml_name declaration.')
        mode, name = declaration.split(':')
        if mode not in MODES or not re.fullmatch(r'TSGgml_[A-Za-z][A-Za-z0-9_]*', name):
            raise ValueError('Invalid mode or exact TensorSharp-owned export name.')
        if mode == 'production' and 'Test' in name:
            raise ValueError('Production added exports may not expose test hooks.')
        if name in result[mode]:
            raise ValueError('Duplicate added-export declaration.')
        result[mode].append(name)
    return {mode: sorted(names) for mode, names in result.items()}


def validate_export_set(actual, base, additions, mode):
    if mode not in MODES or not isinstance(additions, list):
        raise ValueError('Require an explicit export mode and symbol list.')
    declared = declared_added_exports([mode + ':' + name for name in additions])[mode]
    actual, base = set(actual), set(base)
    if not actual or not base or set(declared) & base:
        raise ValueError('Added exports must be absent from the complete nonempty pinned base.')
    if actual != base | set(declared):
        raise ValueError('Native exports differ from the exact pinned base plus declared additions.')
    if mode == 'production' and any('TSGgml_' in name and 'Test' in name for name in actual):
        raise ValueError('Production DLL exposes test hooks.')
    return dict(count=len(actual), added=declared, removed=[])


def expected_capabilities(base, glm_adapter_version=None, glm_slot_copy_version=None):
    result = dict(base)
    if glm_adapter_version is not None:
        if type(glm_adapter_version) is not int or not 1 <= glm_adapter_version <= 100:
            raise ValueError('Require an explicit bounded positive GLM adapter version.')
        name = 'TSGgml_GlmSelectedExpertAdapterVersion'
        if name not in result:
            raise ValueError('The fixed base must already expose the GLM adapter capability.')
        result[name] = glm_adapter_version
    if glm_slot_copy_version is not None:
        if type(glm_slot_copy_version) is not int or not 1 <= glm_slot_copy_version <= 100:
            raise ValueError('Require an explicit bounded positive GLM slot-copy version.')
        name = 'TSGgml_GlmSlotCopyVersion'
        if name in result and result[name] != glm_slot_copy_version:
            raise ValueError('An inherited GLM slot-copy capability must remain unchanged.')
        result[name] = glm_slot_copy_version
    return result


def prepare(args):
    if os.name!='nt':raise RuntimeError('This validator requires Windows/MSVC.')
    base_root=Path(args.base).resolve(strict=True);out=Path(args.output).resolve();repo=Path(args.repository).resolve(strict=True)
    if out.exists() or out==base_root or out in base_root.parents or base_root in out.parents:
        raise ValueError('Fresh independent output must not contain or sit inside the immutable base.')
    nodes=build_chain(base_root);base=nodes[0]['Summary']
    export_additions=declared_added_exports(getattr(args, 'allow_added_export', []))
    libraries=cache_value(base_root/'build-hooks/CMakeCache.txt','TENSORSHARP_VALIDATION_LIBRARIES')
    before=live_pins(repo,base,libraries)
    added=set(before['current_sources'])-set(base['source_snapshot'])
    declared={};replacements=next((node['Summary'].get('generated_driver_changes') for node in nodes
                                  if node['Summary'].get('generated_driver_changes')),None)
    if before['current_sources'].get(CMAKE)!=base['source_snapshot'].get(CMAKE):
        declared=declared_test_additions((base_root/'source'/CMAKE).read_text('utf-8'),
                                         (repo/CMAKE).read_text('utf-8'),added)
    changes=validate_live_sources(before['current_sources'],base['source_snapshot'],args.allow_test_overlay,
                                  allow_owned_changes=True,declared_test_configuration=bool(declared))
    closure_changes=sorted(set(before['current_sources'])^set(base['source_snapshot']))
    production_closure=sorted(set(closure_changes)-set(declared.get('SourcePaths',[])))
    affected_changes=sorted(set(changes)-({CMAKE} if declared else set())-set(declared.get('SourcePaths',[])))
    plan=dict(Schema='tensorsharp.owned-cache-incremental-build.v2',BaseRoot=str(base_root),Output=str(out),
              Repository=str(repo),BaseReceipt=pin(base_root/'build-summary.json'),Libraries=libraries,
              PreparationTool=pin(__file__),before=before,source_snapshot=before['current_sources'],
              ChangedOwnedSources=changes,SourceClosureChanges=closure_changes,ExplicitTestOverlays=args.allow_test_overlay,Modes={},
              ProductionSourceClosureChanges=production_closure,DeclaredStandaloneTestAdditions=declared,
              DependencyAffectedChanges=affected_changes,generated_driver_changes=replacements,
              AncestorReceipts=[node['Receipt'] for node in nodes],
              AncestorPlans=[node['PlanPin'] for node in nodes if 'PlanPin' in node],DependencyEvidenceFiles={},
              ExplicitGlmAdapterVersion=getattr(args, 'expected_glm_adapter_version', None),
              ExplicitGlmSlotCopyVersion=getattr(args, 'expected_glm_slot_copy_version', None),
              ExplicitAddedExports=export_additions,
              ExplicitOwnedRecompiles=getattr(args, 'recompile_owned_target', []),
              Scope='Recorded Ninja deps select every affected owned TU. Any proposed reuse lacking valid generation/dependency proof causes all69ownedTUs of that mode to recompile. No ggml compilation, model, GPU graph, quality, throughput or default-readiness acceptance.')
    # Verify all base source snapshots and generated driver files before copying.
    for name,expected in base['source_snapshot'].items():
        if sha(base_root/'source'/name)!=expected:raise ValueError('Base source snapshot changed: '+name)
        relative=Path(name).relative_to('TensorSharp.GGML.Native')
        wanted=base['generated_driver_files'].get('driver/'+relative.as_posix(),expected)
        if sha(base_root/'driver'/relative)!=wanted:raise ValueError('Base generated driver changed: '+name)
    for relative,expected in base['generated_driver_files'].items():
        if sha(base_root/relative)!=expected:raise ValueError('Base generated native input changed: '+relative)
    out.mkdir(parents=True)
    shutil.copytree(base_root/'source',out/'source');shutil.copytree(base_root/'driver',out/'driver')
    shutil.copy2(base_root/'import-upstream.cmake',out/'import-upstream.cmake')
    for name in changes:
        targets=[out/'source'/name,out/'driver'/Path(name).relative_to('TensorSharp.GGML.Native')]
        if name in before['current_sources']:
            for target in targets:
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/name,target)
        else:
            for target in targets:
                if not target.resolve().is_relative_to(out):raise ValueError('Fresh snapshot removal escaped output scope.')
                target.unlink()
    if declared:
        driver=regenerate_driver_cmake((repo/CMAKE).read_text('utf-8'),replacements,
                                      (base_root/'source'/CMAKE).read_text('utf-8'),
                                      (base_root/'driver/CMakeLists.txt').read_text('utf-8'))
        (out/'driver/CMakeLists.txt').write_text(driver,'utf-8',newline='')
    plan['generated_driver_files']={relative:sha(out/relative) for relative in base['generated_driver_files']}
    plan['FrozenDriverFiles']={str(path.relative_to(out)).replace('\\','/'):sha(path)
                               for path in (out/'driver').rglob('*') if path.is_file()}
    for mode in MODES:
        old=base_root/('build-'+mode);new=out/('build-'+mode);new.mkdir()
        (new/'CMakeFiles/GgmlOps.dir').mkdir(parents=True)
        provenance=object_provenance(nodes,mode);objects=base[mode+'_objects']
        configured,rows,database_pin,commands_pin=command_rows(nodes,mode)
        if set(rows)!=set(objects):raise ValueError('Configured/base object target closure differs.')
        configured_build=configured['Root']/('build-'+mode)
        lines=Path(commands_pin['Path']).read_text('utf-8-sig').splitlines();final=lines[-1]
        ninja=cache_value(configured_build/'CMakeCache.txt','CMAKE_MAKE_PROGRAM')
        deps_by_root={};units={}
        for name,item in provenance.items():
            row=rows[name];source='TensorSharp.GGML.Native/'+Path(row['file']).relative_to(configured['Root']/'driver').as_posix()
            if source not in before['current_sources']:raise ValueError('A compiled translation unit was removed; configure a fresh full native target.')
            origin=item['Origin'];key=str(origin['Root'])
            # An explicitly changed translation unit is compiled regardless of
            # missing old deps. Only proposed reuse requires dependency proof.
            proof=dict(Valid=False,Reason='translation-unit-changed')
            if source not in affected_changes:
                if key not in deps_by_root:
                    deps_by_root[key]=recorded_dependencies(origin,mode,ninja,plan['DependencyEvidenceFiles'])
                records=deps_by_root[key]
                record=records.get(canonical(name)) if records else None
                proof=prove_owned_dependencies(record,origin,mode,base['source_snapshot'],source,repo)
            units[name]=dict(Source=source,ObjectPath=item['Path'],ObjectSha256=item['Sha256'],
                            OriginRoot=str(origin['Root']),DependencyProof=proof)
        decisions=rebuild_decisions(units,affected_changes,args.force_all_owned or bool(production_closure),
                                   getattr(args, 'recompile_owned_target', []))
        if production_closure:
            for decision in decisions.values():decision.update(Reason='owned-source-closure-changed-full-owned-recompile',ClosureChanges=production_closure)
        compile_commands=[];final_objects={};new_objects={}
        for name,unit in units.items():
            if decisions[name]['Rebuild']:
                row=rows[name]
                # The configured ancestor records flags; base incremental
                # receipts assert those flags only moved sources/outputs.
                argv,destination=rewrite_owned_compile(row,configured_build,new,configured['Root']/'driver',out/'driver')
                destination.parent.mkdir(parents=True,exist_ok=True)
                compile_commands.append(dict(Target=name,Source=unit['Source'],Command=argv,
                                             Output=str(destination),Reason=decisions[name]))
                final_objects[name]=str(destination);new_objects[name]=str(destination)
            else:final_objects[name]=unit['ObjectPath']
        definition=old/'ggml_api_exports.def'
        if not definition.is_file():raise ValueError('Missing exact base linker export definition.')
        shutil.copy2(definition,new/'ggml_api_exports.def')
        # Metadata-only cache is copied for component tooling to read imported
        # library/tool locations. This directory is NOT a configured Ninja tree.
        shutil.copy2(old/'CMakeCache.txt',new/'CMakeCache.txt')
        link_argv=rewrite_link(final,configured_build,new,objects,None,new/'ggml_api_exports.def',object_paths=final_objects)
        linker=next((Path(value) for value in link_argv if Path(value).name.lower()=='link.exe'),None)
        if linker is None:raise ValueError('Missing pinned MSVC linker.')
        dumpbin=str(linker.with_name('dumpbin.exe'))
        base_symbols=symbols(subprocess.check_output([dumpbin,'/exports',str(old/'GgmlOps.dll')],text=True,timeout=60))
        validate_export_set(base_symbols | set(export_additions[mode]),base_symbols,export_additions[mode],mode)
        slot_copy_version=getattr(args, 'expected_glm_slot_copy_version', None)
        if slot_copy_version is not None and 'TSGgml_GlmSlotCopyVersion' not in base_symbols | set(export_additions[mode]):
            raise ValueError('An explicit slot-copy capability requires its exact declared or inherited export in both modes.')
        plan['Modes'][mode]=dict(BaseBuild=str(old),NewBuild=str(new),BaseObjects=objects,
            CompileDatabase=database_pin,NinjaCommands=commands_pin,BaseExportDefinition=pin(definition),
            NewExportDefinition=pin(new/'ggml_api_exports.def'),BaseCMakeCache=pin(old/'CMakeCache.txt'),
            NewCMakeCache=pin(new/'CMakeCache.txt'),CompileCommands=compile_commands,LinkCommand=link_argv,
            BaseObjectProvenance={name:{key:value for key,value in item.items() if key!='Origin'} for name,item in provenance.items()},
            Units=units,RebuildDecisions=decisions,FinalObjectPaths=final_objects,NewObjects=new_objects,
            RecompiledCount=len(new_objects),ReusedCount=len(objects)-len(new_objects),Dumpbin=dumpbin,
            BaseExportSymbols=sorted(base_symbols),AllowedAddedExports=export_additions[mode],
            ExpectedCapabilities=expected_capabilities(base[mode+'_constant_capability_queries'],
                getattr(args, 'expected_glm_adapter_version', None),slot_copy_version),BaseArtifacts=base[mode+'_artifacts'])
    save(out/'incremental-plan.json',plan)
    print(json.dumps(dict(Prepared=True,ChangedOwnedSources=changes,
                         Counts={mode:dict(Recompiled=row['RecompiledCount'],Reused=row['ReusedCount']) for mode,row in plan['Modes'].items()},Output=str(out))))


def verify_plan(plan):
    if pin(__file__)!=plan['PreparationTool'] or pin(plan['BaseReceipt']['Path'])!=plan['BaseReceipt']:
        raise ValueError('Frozen tool/base receipt changed.')
    base=read(plan['BaseReceipt']['Path'])
    for row in plan.get('AncestorReceipts',[])+plan.get('AncestorPlans',[]):
        if pin(row['Path'])!=row:raise ValueError('Pinned complete ancestor provenance changed.')
    for row in plan.get('DependencyEvidenceFiles',{}).values():
        if pin(row['Path'])!=row:raise ValueError('Recorded Ninja dependency metadata changed.')
    live=live_pins(plan['Repository'],base,plan['Libraries'])
    if live!=plan['before']:raise ValueError('Live sources/upstream/library closure changed during the incremental build.')
    out=Path(plan['Output'])
    if {name:sha(out/'source'/name) for name in plan['source_snapshot']}!=plan['source_snapshot']:
        raise ValueError('Effective source snapshot changed.')
    if {str(path.relative_to(out)).replace('\\','/'):sha(path) for path in (out/'driver').rglob('*') if path.is_file()}!=plan['FrozenDriverFiles']:
        raise ValueError('Frozen driver/header closure changed.')
    for mode,entry in plan['Modes'].items():
        if 'BaseObjectProvenance' in entry:
            if set(entry['BaseObjectProvenance'])!=set(entry['BaseObjects']):raise ValueError('Incomplete base object provenance.')
            for name,row in entry['BaseObjectProvenance'].items():
                if row['Sha256']!=entry['BaseObjects'][name] or sha(row['Path'])!=row['Sha256']:
                    raise ValueError('Pinned inherited/recompiled base object changed: '+name)
        else:check_object_closure(Path(entry['BaseBuild']),entry['BaseObjects'])
        for key in ('CompileDatabase','NinjaCommands','BaseExportDefinition','NewExportDefinition','BaseCMakeCache','NewCMakeCache'):
            if pin(entry[key]['Path'])!=entry[key]:raise ValueError('Pinned base/new native input changed: '+key)
        for name,expected in entry['BaseArtifacts'].items():
            if sha(Path(entry['BaseBuild'])/name)!=expected:raise ValueError('Immutable base native artifact changed.')
    return live


def symbols(exports):return set(re.findall(r'^\s+\d+\s+[A-F0-9]+\s+[A-F0-9]+\s+(\S+)',exports,re.M))


def run(args):
    if os.name!='nt' or os.environ.get('CUDA_VISIBLE_DEVICES')!='-1':
        raise RuntimeError('Run in a configured Windows VC environment with CUDA_VISIBLE_DEVICES=-1; no GPU execution.')
    plan=read(args.plan);out=Path(plan['Output']);summary=out/'build-summary.json'
    if summary.exists():raise ValueError('Never overwrite or resume a completed/failed build receipt.')
    report=dict(utc_start=utc(),passed=False,no_upstream_compilation=True,fresh_all_owned_objects=False,
                ReusedOwnedObjectsPerMode={mode:entry.get('ReusedCount',68) for mode,entry in plan['Modes'].items()},
                RecompiledOwnedObjectsPerMode={mode:entry.get('RecompiledCount',1) for mode,entry in plan['Modes'].items()},steps=[],
                IncrementalPlan=pin(args.plan),validation_scope=plan['Scope'],
                source_snapshot=plan['source_snapshot'],generated_driver_files=plan['generated_driver_files'],
                generated_driver_changes=plan.get('generated_driver_changes'))
    def child(label,command,cwd):
        started=time.monotonic();log=out/(label+'.log')
        with log.open('xb') as stream:
            process=subprocess.Popen(command,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,shell=False)
            try:code=process.wait(timeout=args.timeout)
            except BaseException:
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=stream,stderr=subprocess.STDOUT,
                               timeout=30,shell=False)
                process.wait(timeout=30)
                raise
        report['steps'].append(dict(name=label,command=command,cwd=str(cwd),pid=process.pid,exit_code=code,
                                    seconds=time.monotonic()-started,log_sha256=sha(log)))
        save(out/'progress.json',report)
        if code:raise RuntimeError(label+' failed with exit '+str(code))
    try:
        report['before']=verify_plan(plan)
        for mode,entry in plan['Modes'].items():
            build=Path(entry['NewBuild'])
            compiled=entry.get('CompileCommands')
            if compiled is None:compiled=[dict(Target='CMakeFiles/GgmlOps.dir/ggml_ops_host_moe_cache.cpp.obj',
                                               Command=entry['CompileCommand'],Output=entry['NewCacheObject'])]
            if any(Path(row['Output']).exists() for row in compiled) or (build/'GgmlOps.dll').exists():
                raise ValueError('Fresh incremental compile/link outputs required.')
            for index,row in enumerate(compiled):
                child('compile-'+mode+'-'+str(index+1),row['Command'],build)
                if not Path(row['Output']).is_file():raise ValueError('Compiler did not emit the selected owned object.')
            child('link-'+mode,entry['LinkCommand'],build)
            report[mode+'_artifacts']={path.name:sha(path) for path in build.iterdir()
                                      if path.is_file() and path.suffix.lower() in ('.dll','.lib','.exp')}
            if set(report[mode+'_artifacts'])!={'GgmlOps.dll','GgmlOps.lib','GgmlOps.exp'}:
                raise ValueError('Incomplete fresh DLL/link artifacts.')
            final=entry.get('FinalObjectPaths') or {name:(entry['NewCacheObject'] if name.endswith('ggml_ops_host_moe_cache.cpp.obj')
                                                        else str(Path(entry['BaseBuild'])/name)) for name in entry['BaseObjects']}
            report[mode+'_objects']={name:sha(path) for name,path in final.items()}
            new_targets={row['Target'] for row in compiled}
            report[mode+'_object_provenance']={name:dict(Path=path,Sha256=report[mode+'_objects'][name],Recompiled=name in new_targets,
                                                       Reason=entry.get('RebuildDecisions',{}).get(name)) for name,path in final.items()}
            report[mode+'_compiler_comparison']=dict(reused_unchanged_objects=len(final)-len(compiled),recompiled_owned_objects=len(compiled),
                                                       original_compile_database_sha256=entry['CompileDatabase']['Sha256'],
                                                       unchanged_compiler_options_except_source_and_outputs=True,
                                                       unchanged_linker_options_except_objects_and_fresh_outputs=True)
            exports=subprocess.check_output([entry['Dumpbin'],'/exports',str(build/'GgmlOps.dll')],text=True,timeout=60)
            old=subprocess.check_output([entry['Dumpbin'],'/exports',str(Path(entry['BaseBuild'])/'GgmlOps.dll')],text=True,timeout=60)
            (out/(mode+'-exports.log')).write_text(exports,'utf-8')
            if 'BaseExportSymbols' in entry and sorted(symbols(old))!=entry['BaseExportSymbols']:
                raise ValueError('The pinned base export symbols changed after preparation.')
            report[mode+'_exports']=validate_export_set(symbols(exports),symbols(old),entry.get('AllowedAddedExports',[]),mode)
            code='import ctypes,json,os,sys; h=os.add_dll_directory(sys.argv[2]); l=ctypes.CDLL(sys.argv[1]); print(json.dumps({n:getattr(l,n)() for n in json.loads(sys.argv[3])}))'
            actual=json.loads(subprocess.check_output([sys.executable,'-B','-c',code,str(build/'GgmlOps.dll'),
                args.cuda_bin,json.dumps(entry['ExpectedCapabilities'])],text=True,timeout=60))
            if actual!=entry['ExpectedCapabilities']:raise ValueError('Native constant capabilities differ; no model is invoked.')
            report[mode+'_constant_capability_queries']=actual
        report['after']=verify_plan(plan)
        if report['after']!=report['before'] or pin(args.plan)!=report['IncrementalPlan']:
            raise ValueError('Native build input bookends changed.')
        report['passed']=True
    except BaseException as error:
        report['error']=repr(error);report['traceback']=traceback.format_exc()
    finally:
        report['utc_end']=utc();save(summary,report)
    print(json.dumps(dict(passed=report['passed'],error=report.get('error'),summary=str(summary))))
    return 0 if report['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--base',required=True);prep.add_argument('--output',required=True)
    prep.add_argument('--repository',default='C:/Works/TensorSharp');prep.add_argument('--allow-test-overlay',action='append',default=[])
    prep.add_argument('--force-all-owned',action='store_true',help='Explicit69-TU/modefallback; ggml still stays prebuilt and unchanged.')
    prep.add_argument('--expected-glm-adapter-version',type=int,
                      help='Explicit expected owned ABI change; all other capabilities retain the pinned base values and actual DLL queries must match exactly.')
    prep.add_argument('--allow-added-export',action='append',default=[],
                      help='Exact mode:TSGgml_name added symbol. Defaults retain exact base exports; no removals or unlisted additions are accepted.')
    prep.add_argument('--expected-glm-slot-copy-version',type=int,
                      help='Explicit new slot-copy constant query. Its exact export must be inherited or declared in both modes; all inherited capability values remain pinned.')
    prep.add_argument('--recompile-owned-target',action='append',default=[],
                      help='Exact recorded owned object target to rebuild. This cannot authorize unproven reuse; all other insufficient dependency proofs still force a complete owned rebuild.')
    execute=sub.add_parser('run');execute.add_argument('--plan',required=True);execute.add_argument('--timeout',type=int,default=600)
    execute.add_argument('--cuda-bin',default='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin')
    args=parser.parse_args()
    if args.action=='prepare':prepare(args);return 0
    if not 1<=args.timeout<=2400:parser.error('Require a bounded per-child native timeout.')
    return run(args)


if __name__=='__main__':raise SystemExit(main())
