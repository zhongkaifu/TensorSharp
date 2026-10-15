import importlib.util
import copy
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SPEC=importlib.util.spec_from_file_location('owned_native_cache_incremental_test',
    Path(__file__).parents[1]/'rebuild-owned-native-cache.py')
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


class ImmutableClosureTests(unittest.TestCase):
    def test_actual_nvcc_make_paths_drive_separators_spaces_and_continuations(self):
        target=r'E:\frozen build\obj\a.cu.obj';source='E:/frozen build/driver/a.cu'
        text=(r'E:\frozen\ build\obj\a.cu.obj : E:/frozen\ build/driver/a.cu ' + '\\\n'
              r' E:/frozen\ build/driver/a.cu C:/Program\ Files/SDK/header.h' + '\n')
        paths=MODULE.parse_nvcc_dependencies(text,target,source)
        self.assertEqual([r'E:\frozen build\driver\a.cu',r'C:\Program Files\SDK\header.h'],paths)
        escaped=r'E\:/obj/a.cu.obj : E\:/driver/a.cu'
        self.assertEqual([r'E:\driver\a.cu'],MODULE.parse_nvcc_dependencies(escaped,'E:/obj/a.cu.obj','E:/driver/a.cu'))
        self.assertEqual([r'E:\driver\a.cu',r'\\server\share\header.h'],MODULE.parse_nvcc_dependencies(
            r'E:/obj/a.cu.obj : E:/driver/a.cu \\server\share\header.h','E:/obj/a.cu.obj','E:/driver/a.cu'))

    def test_nvcc_rejects_unknown_rules_macros_targets_and_incomplete_evidence(self):
        target='E:/obj/a.cu.obj';source='E:/driver/a.cu';valid=target+' : '+source
        for bad in [valid.replace(target,'E:/obj/b.cu.obj'),target+' : E:/driver/other.cu',
                    target+' E:/obj/b.obj : '+source,valid+'\nE:/other.obj : '+source,
                    valid+' $(EXTRA)',valid+' ${EXTRA}',valid+' \\$(EXTRA)',valid+' # comment',
                    valid+' relative.h',valid+' bad\\q.h',valid+' \\',valid+'\0',
                    valid+' E:/SDK/b.h : E:/SDK/c.h',valid.replace(' : ',' '),valid+' "E:/quoted.h"']:
            with self.subTest(bad=bad),self.assertRaises(ValueError):MODULE.parse_nvcc_dependencies(bad,target,source)

    def test_cl_includes_require_literal_actual_markers(self):
        source='E:/driver/a.cpp'
        text='a.cpp\nNote: including file:   E:\\driver\\owned.h\nNote: including file: C:\\SDK\\vendor.h\n'
        self.assertEqual([r'E:\driver\a.cpp',r'E:\driver\owned.h',r'C:\SDK\vendor.h'],MODULE.parse_cl_dependencies(text,source))
        for bad in ['a.cpp\n','Note: including file:\n','Note: including file: ../owned.h\n',
                    text+'x Note: including file: E:/unsafe.h\n',text+'Note: including file: $(HEADERS)\n',text+'\0']:
            with self.subTest(bad=bad),self.assertRaises(ValueError):MODULE.parse_cl_dependencies(bad,source)

    def test_capabilities_inherit_exact_base_unless_explicit_adapter_change(self):
        original = {'TSGgml_GlmSelectedExpertAdapterVersion': 6, 'OtherPinnedCapability': 4}
        self.assertEqual(original, MODULE.expected_capabilities(original))
        self.assertIsNot(original, MODULE.expected_capabilities(original))
        self.assertEqual(original | {'TSGgml_GlmSelectedExpertAdapterVersion': 7},
                         MODULE.expected_capabilities(original, 7))
        self.assertEqual(6, original['TSGgml_GlmSelectedExpertAdapterVersion'])
        for bad in (0, -1, 101, True, '7'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                MODULE.expected_capabilities(original, bad)
        with self.assertRaises(ValueError): MODULE.expected_capabilities({'OtherPinnedCapability': 4}, 7)

    def test_additive_exports_require_exact_explicit_mode_and_symbol(self):
        names = ['TSGgml_GlmSlotBytes', 'TSGgml_GlmSlotCopyInto', 'TSGgml_GlmSlotCopyVersion']
        hooks = ['TSGgml_GlmTestSlotAllocFault', 'TSGgml_GlmTestSlotCompare', 'TSGgml_GlmTestSlotCopyFault']
        declarations = ['production:' + name for name in names] + ['hooks:' + name for name in names + hooks]
        expected = dict(production=sorted(names), hooks=sorted(names + hooks))
        self.assertEqual(expected, MODULE.declared_added_exports(declarations))
        self.assertEqual(dict(production=[], hooks=[]), MODULE.declared_added_exports([]))
        for bad in (['other:TSGgml_GlmSlotBytes'], ['production:TSGgml_GlmTestSlotCompare'],
                    ['hooks:ggml_upstream'], ['hooks:TSGgml_*'], ['hooks:TSGgml_Name:Extra'],
                    ['hooks:TSGgml_Name\n'], ['hooks:TSGgml_Name;command'],
                    ['hooks:TSGgml_Name'] * 2, [None], None,
                    ['hooks:TSGgml_Name' + str(i) for i in range(33)]):
            with self.subTest(bad=bad), self.assertRaises(ValueError): MODULE.declared_added_exports(bad)

    def test_default_export_validation_preserves_complete_base_exactness(self):
        base = {'ggml_unmodified_export', 'TSGgml_GlmForward'}
        self.assertEqual(dict(count=2, added=[], removed=[]), MODULE.validate_export_set(base, base, [], 'production'))
        for actual in (base | {'TSGgml_Unexpected'}, base - {'TSGgml_GlmForward'}, set()):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                MODULE.validate_export_set(actual, base, [], 'production')

    def test_added_export_validation_refuses_removals_missing_extra_and_already_base(self):
        base = {'ggml_unmodified_export', 'TSGgml_GlmForward'}
        extra = ['TSGgml_GlmSlotBytes']
        self.assertEqual(dict(count=3, added=extra, removed=[]),
            MODULE.validate_export_set(base | set(extra), base, extra, 'production'))
        for actual, additions, mode in (
            (base, extra, 'production'), (base | set(extra) | {'TSGgml_Unlisted'}, extra, 'production'),
            ((base - {'TSGgml_GlmForward'}) | set(extra), extra, 'production'),
            (base, ['TSGgml_GlmForward'], 'production'),
            (base | set(extra), extra * 2, 'production'),
            (base | {'TSGgml_GlmTestSlotCompare'}, ['TSGgml_GlmTestSlotCompare'], 'production'),
            (base | set(extra), extra, 'unknown'),
            (base | set(extra), [], 'hooks'), (base | set(extra), extra, 'Production'),
        ):
            with self.subTest(actual=actual, additions=additions, mode=mode), self.assertRaises(ValueError):
                MODULE.validate_export_set(actual, base, additions, mode)

    def test_new_slot_copy_constant_query_is_bounded_and_inherited_values_stay_pinned(self):
        base = {'TSGgml_GlmSelectedExpertAdapterVersion': 7, 'OtherPinnedCapability': 4}
        self.assertEqual(base | {'TSGgml_GlmSlotCopyVersion': 1}, MODULE.expected_capabilities(base, 7, 1))
        self.assertEqual(base, MODULE.expected_capabilities(base))
        self.assertNotIn('TSGgml_GlmSlotCopyVersion', base)
        inherited = base | {'TSGgml_GlmSlotCopyVersion': 1}
        self.assertEqual(inherited, MODULE.expected_capabilities(inherited, None, 1))
        for bad in (0, -1, 101, True, '1', 2):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                MODULE.expected_capabilities(inherited, None, bad)

    def test_changed_header_or_other_translation_unit_refuses_reuse(self):
        base={MODULE.CACHE:'old','TensorSharp.GGML.Native/owned.h':'same',
              'TensorSharp.GGML.Native/tests/fixture.inc':'fixture'}
        current=dict(base);current[MODULE.CACHE]='new'
        self.assertEqual(MODULE.validate_live_sources(current,base,[]),[MODULE.CACHE])
        current['TensorSharp.GGML.Native/owned.h']='changed'
        with self.assertRaises(ValueError):MODULE.validate_live_sources(current,base,[])

    def test_only_explicit_test_fixtures_can_be_overlaid(self):
        name='TensorSharp.GGML.Native/tests/fixture.inc'
        base={MODULE.CACHE:'old',name:'fixture'};current=dict(base);current[name]='new'
        with self.assertRaises(ValueError):MODULE.validate_live_sources(current,base,[])
        self.assertEqual(MODULE.validate_live_sources(current,base,[name]),[name])
        for overlay in ('TensorSharp.GGML.Native/owned.h','../tests/fixture.inc',
                        'TensorSharp.GGML.Native/tests/unknown.inc'):
            with self.subTest(overlay=overlay),self.assertRaises(ValueError):
                MODULE.validate_live_sources(current,base,[overlay])

    def test_missing_or_mutated_object_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            build=Path(directory);obj=build/'CMakeFiles/GgmlOps.dir';obj.mkdir(parents=True)
            for index in range(69):(obj/f'{index}.obj').write_bytes(bytes([index]))
            expected={str(path.relative_to(build)).replace('\\','/'):MODULE.sha(path) for path in obj.iterdir()}
            self.assertEqual(MODULE.check_object_closure(build,expected),expected)
            (obj/'0.obj').write_bytes(b'changed')
            with self.assertRaises(ValueError):MODULE.check_object_closure(build,expected)
            (obj/'0.obj').unlink()
            with self.assertRaises(ValueError):MODULE.check_object_closure(build,expected)

    def test_shell_wrapper_cannot_smuggle_an_extra_command(self):
        valid='C:\\Windows\\system32\\cmd.exe /C "cd . && "C:\\Program Files\\CMake\\bin\\cmake.exe" -E vs_link_dll -- link.exe && cd ."'
        self.assertTrue(MODULE.strip_link_shell(valid).startswith('"C:\\Program Files'))
        with self.assertRaises(ValueError):MODULE.strip_link_shell(valid.replace(' -- link.exe',' && echo changed -- link.exe'))
        with self.assertRaises(ValueError):MODULE.strip_link_shell('link.exe arbitrary')

    def test_recorded_dependency_counts_and_status_are_preserved(self):
        text='CMakeFiles/GgmlOps.dir/a.cpp.obj: #deps 2, deps mtime 100 (VALID)\n    ../driver/owned.h\n    C:/sdk/header.h\n\n'
        value=MODULE.parse_ninja_dependencies(text)['cmakefiles/ggmlops.dir/a.cpp.obj']
        self.assertTrue(value['Valid']);self.assertEqual(value['Dependencies'],['../driver/owned.h','C:/sdk/header.h'])
        self.assertFalse(MODULE.parse_ninja_dependencies(text.replace('(VALID)','(STALE)'))['cmakefiles/ggmlops.dir/a.cpp.obj']['Valid'])
        for invalid in (text.replace('#deps 2','#deps 3'),text+text,'a.obj: deps not found.'):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):MODULE.parse_ninja_dependencies(invalid)

    def test_header_change_rebuilds_every_recorded_dependent_tu(self):
        header='TensorSharp.GGML.Native/ExpertFileSource.h'
        units={name:dict(Source='TensorSharp.GGML.Native/'+name,DependencyProof=dict(Valid=True,OwnedDependencies=deps))
               for name,deps in [('cache.cpp',[header]),('moe.cpp',[header]),('other.cpp',[])]}
        result=MODULE.rebuild_decisions(units,[header])
        self.assertTrue(result['cache.cpp']['Rebuild']);self.assertTrue(result['moe.cpp']['Rebuild'])
        self.assertFalse(result['other.cpp']['Rebuild'])

    def test_unknown_reuse_forces_complete_mode_not_guessed_include(self):
        units={'known.cpp':dict(Source='known.cpp',DependencyProof=dict(Valid=True,OwnedDependencies=[])),
               'unknown.cu':dict(Source='unknown.cu',DependencyProof=dict(Valid=False))}
        result=MODULE.rebuild_decisions(units,['a.h'])
        self.assertTrue(all(row['Rebuild'] for row in result.values()))
        self.assertEqual(result['known.cpp']['UnprovenReuseTargets'],['unknown.cu'])
        self.assertIn('full-owned',result['known.cpp']['Reason'])

    def test_explicit_recompile_can_avoid_reusing_an_unproven_owned_object(self):
        units={'known.cpp':dict(Source='known.cpp',DependencyProof=dict(Valid=True,OwnedDependencies=[])),
               'unknown.cu':dict(Source='unknown.cu',DependencyProof=dict(Valid=False))}
        result=MODULE.rebuild_decisions(units,[],False,['unknown.cu'])
        self.assertFalse(result['known.cpp']['Rebuild'])
        self.assertTrue(result['unknown.cu']['Rebuild'])
        self.assertEqual('explicit-owned-target-recompile',result['unknown.cu']['Reason'])
        for forced in (['missing.cpp'], ['unknown.cu','unknown.cu']):
            with self.subTest(forced=forced), self.assertRaises(ValueError):
                MODULE.rebuild_decisions(units,[],False,forced)
        units['another.cu']=dict(Source='another.cu',DependencyProof=dict(Valid=False))
        self.assertTrue(all(row['Rebuild'] for row in MODULE.rebuild_decisions(units,[],False,['unknown.cu']).values()))

    def test_explicit_changed_tu_does_not_need_reuse_dependency_proof(self):
        units={'cache.cpp':dict(Source='cache.cpp',DependencyProof=dict(Valid=False)),
               'other.cpp':dict(Source='other.cpp',DependencyProof=dict(Valid=True,OwnedDependencies=[]))}
        result=MODULE.rebuild_decisions(units,['cache.cpp'])
        self.assertEqual(result['cache.cpp']['Reason'],'owned-translation-unit-changed')
        self.assertFalse(result['other.cpp']['Rebuild'])
        self.assertTrue(all(row['Rebuild'] for row in MODULE.rebuild_decisions(units,[],True).values()))

    def test_owned_changes_allow_header_but_not_target_configuration(self):
        base={MODULE.CACHE:'same','TensorSharp.GGML.Native/owned.h':'old','TensorSharp.GGML.Native/CMakeLists.txt':'cfg'}
        current=dict(base);current['TensorSharp.GGML.Native/owned.h']='new'
        self.assertEqual(MODULE.validate_live_sources(current,base,[],True),['TensorSharp.GGML.Native/owned.h'])
        current['TensorSharp.GGML.Native/CMakeLists.txt']='different'
        with self.assertRaises(ValueError):MODULE.validate_live_sources(current,base,[],True)

    def test_new_standalone_tests_preserve_production_configuration(self):
        before='add_library(GgmlOps SHARED owned.cpp)\nif (TENSORSHARP_GGML_NATIVE_BUILD_TESTS)\n    # existing tests\nendif()\n'
        addition=('    add_executable(NewTest tests/new_test.cpp)\n'
                  '    target_include_directories(NewTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")\n'
                  '    add_test(NAME new-test COMMAND NewTest)\n'
                  '    set_tests_properties(new-test PROPERTIES SKIP_RETURN_CODE 77)\n')
        after=before.replace('endif()',addition+'endif()')
        source='TensorSharp.GGML.Native/tests/new_test.cpp'
        value=MODULE.declared_test_additions(before,after,{source})
        self.assertEqual(value['SourcePaths'],[source]);self.assertEqual(value['Tests'],{'new-test':'NewTest'})
        for bad in (after.replace('owned.cpp','other.cpp'),after.replace('SKIP_RETURN_CODE 77','SKIP_RETURN_CODE 0'),
                    after.replace('tests/new_test.cpp','tests/../owned.cpp'),
                    after.replace('add_test(NAME new-test COMMAND NewTest)','target_compile_definitions(GgmlOps PRIVATE CHANGED=1)'),
                    before+addition,after.replace('if (TENSORSHARP_GGML_NATIVE_BUILD_TESTS)','if (WIN32)')):
            with self.subTest(bad=bad),self.assertRaises(ValueError):MODULE.declared_test_additions(before,bad,{source})

    def test_driver_transform_must_reproduce_base_exactly(self):
        original='upstream-anchor\nbase flags\n';driver='pinned-import\nbase flags\n'
        self.assertEqual(MODULE.regenerate_driver_cmake(original+'new test\n',{'upstream-anchor':'pinned-import'},original,driver),
                         driver+'new test\n')
        with self.assertRaises(ValueError):MODULE.regenerate_driver_cmake(original,{},original,driver)
        with self.assertRaises(ValueError):MODULE.regenerate_driver_cmake(original,{'upstream-anchor':'different'},original,driver)


@unittest.skipUnless(os.name=='nt','CommandLineToArgvW/Windows output semantics require Windows')
class WindowsCommandRewriteTests(unittest.TestCase):
    def compiler_origin(self,root,compiler):
        mode='production';build=root/'build-production';driver=root/'driver';build.mkdir();driver.mkdir()
        name='owned.cu' if compiler=='nvcc.exe' else 'owned.cpp';source=driver/name;source.write_text('owned generation')
        header=driver/'owned.h';header.write_text('owned header generation')
        target='CMakeFiles/GgmlOps.dir/'+name+'.obj';output=build/target;output.parent.mkdir(parents=True);output.write_bytes(b'qualified object')
        relative='TensorSharp.GGML.Native/'+name;source_snapshot={relative:MODULE.sha(source),'TensorSharp.GGML.Native/owned.h':MODULE.sha(header)}
        argv=[compiler,'-c',str(source)]
        if compiler=='nvcc.exe':argv+=['-MD','-MT',str(output),'-MF',str(output)+'.d','-o',str(output)]
        else:argv+=['/showIncludes','/Fo'+str(output)]
        plan={'Output':str(root),'source_snapshot':source_snapshot,'before':{'same':1},'Modes':{mode:{'NewBuild':str(build),'CompileCommands':[{
            'Target':target,'Source':relative,'Command':argv,'Output':str(output)}]}}}
        plan_path=root/'incremental-plan.json';MODULE.save(plan_path,plan)
        log=root/'compile-production-1.log'
        if compiler=='nvcc.exe':
            log.write_text('NVCC successful\n');dep=Path(str(output)+'.d')
            escape=lambda value:str(value).replace(' ','\\ ')
            dep.write_text(escape(output)+' : '+escape(source)+' \\\n '+escape(header)+'\n')
        else:log.write_text(name+'\nNote: including file: '+str(header)+'\n')
        output.write_bytes(b'qualified object') # NVCC emits deps before its final object, as the real compiler does.
        summary=dict(passed=True,no_upstream_compilation=True,before={'same':1},after={'same':1},
            IncrementalPlan=MODULE.pin(plan_path),source_snapshot=source_snapshot,generated_driver_files={},
            production_objects={target:MODULE.sha(output)},production_object_provenance={target:{
                'Recompiled':True,'Path':str(output),'Sha256':MODULE.sha(output)}},steps=[{
                    'name':'compile-production-1','command':argv,'cwd':str(build),'exit_code':0,'log_sha256':MODULE.sha(log)}])
        MODULE.save(root/'build-summary.json',summary)
        return dict(Root=root,Summary=summary),target,source,header,log,plan_path

    def test_compiler_fallback_binds_complete_real_generation_and_pins_all_evidence(self):
        for compiler in ('cl.exe','nvcc.exe'):
            with self.subTest(compiler=compiler),tempfile.TemporaryDirectory() as directory:
                origin,target,source,header,log,plan_path=self.compiler_origin(Path(directory),compiler)
                evidence={};records=MODULE.recorded_compiler_dependencies(origin,'production',evidence)
                record=records[MODULE.canonical(target)]
                self.assertTrue(record['Valid']);self.assertEqual([str(source),str(header)],record['Dependencies'])
                self.assertIn(str(log.resolve()),evidence);self.assertIn(str(plan_path.resolve()),evidence)
                self.assertIn(str((origin['Root']/'build-summary.json').resolve()),evidence)
                proof=MODULE.prove_owned_dependencies(record,origin,'production',origin['Summary']['source_snapshot'],
                    'TensorSharp.GGML.Native/'+source.name,Path(directory)/'repo')
                self.assertTrue(proof['Valid']);self.assertIn('TensorSharp.GGML.Native/owned.h',proof['OwnedDependencies'])
                self.assertIn('CompilerEvidence',proof)
                self.assertEqual(records,MODULE.recorded_dependencies(origin,'production','unused-ninja',{}))

    def test_compiler_fallback_rejects_tampered_log_object_command_and_failed_generation(self):
        for corruption in ('log','object','command','failed','source','provenance','dep-target'):
            with self.subTest(corruption=corruption),tempfile.TemporaryDirectory() as directory:
                origin,target,source,header,log,plan_path=self.compiler_origin(Path(directory),'nvcc.exe')
                if corruption=='log':log.write_text('changed output')
                elif corruption=='object':(origin['Root']/'build-production'/target).write_bytes(b'changed')
                elif corruption=='command':origin['Summary']['steps'][0]['command']=['nvcc.exe','different-source.cu']
                elif corruption=='failed':origin['Summary']['steps'][0]['exit_code']=1
                elif corruption=='source':source.write_text('changed source')
                elif corruption=='provenance':origin['Summary']['production_object_provenance'][target]['Recompiled']=False
                elif corruption=='dep-target':Path(str(origin['Root']/'build-production'/target)+'.d').write_text('E:/wrong.obj : '+str(source))
                with self.assertRaises(ValueError):MODULE.recorded_compiler_dependencies(origin,'production',{})
                self.assertIsNone(MODULE.recorded_dependencies(origin,'production','unused-ninja',{}))

    def test_compiler_dependency_header_generation_cannot_hide_source_change(self):
        with tempfile.TemporaryDirectory() as directory:
            origin,target,source,header,log,plan_path=self.compiler_origin(Path(directory),'cl.exe')
            records=MODULE.recorded_compiler_dependencies(origin,'production',{})
            base=copy.deepcopy(origin['Summary']['source_snapshot']);base['TensorSharp.GGML.Native/owned.h']='changed-generation'
            proof=MODULE.prove_owned_dependencies(records[MODULE.canonical(target)],origin,'production',base,
                'TensorSharp.GGML.Native/'+source.name,Path(directory)/'repo')
            self.assertFalse(proof['Valid']);self.assertEqual('recorded-header-generation-not-valid-for-base-object',proof['Reason'])
    def test_actual_ninja_dependency_flags_are_proven_against_database(self):
        with tempfile.TemporaryDirectory() as directory:
            build=Path(directory);source=str(build/'driver/owned.cu');output='CMakeFiles/GgmlOps.dir/owned.cu.obj'
            expected=['nvcc.exe','-DTSG=1','-c',source,'-o',output]
            actual=expected[:2]+['-MD','-MT',output,'-MF',output+'.d']+expected[2:]
            row=dict(command=subprocess.list2cmdline(expected),file=source,output=str(build/output))
            command=subprocess.list2cmdline(actual)
            self.assertEqual(MODULE.qualified_ninja_command(row,[command],build),command)
            for bad in (actual+['-DREGRESSION=1'],[value.replace(output+'.d','wrong.d') for value in actual]):
                with self.subTest(bad=bad),self.assertRaises(ValueError):
                    MODULE.qualified_ninja_command(row,[subprocess.list2cmdline(bad)],build)
            with self.assertRaises(ValueError):MODULE.qualified_ninja_command(row,[command,command],build)

    def test_windows_argument_roundtrip_preserves_quoted_define(self):
        values=['C:\\Program Files\\VS\\cl.exe','-DVALUE="C:/path with spaces"','/FdC:\\new build\\obj\\','-c','C:\\new build\\cache.cpp']
        self.assertEqual(MODULE.windows_argv(subprocess.list2cmdline(values)),values)

    def test_cl_outputs_and_source_only_move_to_new_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old=root/'base';new=root/'fresh';driver=root/'base-driver';fresh=root/'fresh-driver'
            source=driver/Path(MODULE.CACHE).name;output=old/'CMakeFiles/GgmlOps.dir/ggml_ops_host_moe_cache.cpp.obj'
            args=['cl.exe','/nologo','-DTSG_GGML_TEST_HOOKS=1','/FoCMakeFiles\\GgmlOps.dir\\ggml_ops_host_moe_cache.cpp.obj',
                  '/FdCMakeFiles\\GgmlOps.dir\\','/FS','-c',str(source)]
            row=dict(command=subprocess.list2cmdline(args),file=str(source),output=str(output))
            rewritten,destination=MODULE.rewrite_compile(row,old,new,driver,fresh)
            self.assertIn('-DTSG_GGML_TEST_HOOKS=1',rewritten)
            self.assertEqual(destination,new/'CMakeFiles/GgmlOps.dir'/output.name)
            self.assertEqual(rewritten[-1],str(fresh/source.name))
            row['command']=subprocess.list2cmdline(args+['/Fo'+str(root/'outside.obj')])
            with self.assertRaises(ValueError):MODULE.rewrite_compile(row,old,new,driver,fresh)

    def test_link_every_object_once_and_all_outputs_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old=root/'base';new=root/'fresh'
            objects={'CMakeFiles/GgmlOps.dir/'+name:'pin' for name in
                     ['ggml_ops_host_moe_cache.cpp.obj']+[f'{index}.cpp.obj' for index in range(68)]}
            args=['C:\\CMake\\cmake.exe','-E','vs_link_dll','--msvc-ver=1944','--intdir=CMakeFiles\\GgmlOps.dir',
                  '--manifests','--','C:\\VS\\link.exe']+list(objects)+['/out:GgmlOps.dll','/implib:GgmlOps.lib',
                  '/pdb:GgmlOps.pdb','/DEF:ggml_api_exports.def','/dll','D:\\imports\\ggml.lib']
            command='C:\\Windows\\system32\\cmd.exe /C "cd . && '+subprocess.list2cmdline(args)+' && cd ."'
            cache=new/'CMakeFiles/GgmlOps.dir/ggml_ops_host_moe_cache.cpp.obj'
            result=MODULE.rewrite_link(command,old,new,objects,cache,new/'ggml_api_exports.def')
            self.assertIn(str(cache),result);self.assertIn(str(old/'CMakeFiles/GgmlOps.dir/0.cpp.obj'),result)
            self.assertIn('/out:'+str(new/'GgmlOps.dll'),result)
            self.assertIn('D:\\imports\\ggml.lib',result)
            for bad in (command.replace('0.cpp.obj','unknown.cpp.obj',1),command.replace('/out:GgmlOps.dll','/out:Wrong.dll')):
                with self.subTest(bad=bad),self.assertRaises(ValueError):
                    MODULE.rewrite_link(bad,old,new,objects,cache,new/'ggml_api_exports.def')

    def test_nvcc_moves_all_output_and_dependency_paths_without_changing_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old=root/'base';new=root/'fresh';driver=root/'driver';fresh=root/'fresh-driver'
            source=driver/'owned.cu';obj='CMakeFiles\\GgmlOps.dir\\owned.cu.obj'
            values=['nvcc.exe','-DTSG_GGML_USE_CUDA=1','--generate-code=arch=compute_86,code=[sm_86]',
                    '-MD','-MT',obj,'-MF',obj+'.d','-x','cu','-c',str(source),'-o',obj,
                    '-Xcompiler=-FdCMakeFiles\\GgmlOps.dir\\,-FS']
            row=dict(command=subprocess.list2cmdline(values),file=str(source),output=str(old/obj))
            result,destination=MODULE.rewrite_owned_compile(row,old,new,driver,fresh)
            self.assertIn('--generate-code=arch=compute_86,code=[sm_86]',result)
            self.assertEqual(result[result.index('-o')+1],str(destination))
            self.assertEqual(result[result.index('-MF')+1],str(destination)+'.d')
            self.assertIn(str(fresh/'owned.cu'),result)
            bad=dict(row,file=str(root/'ExternalProjects/ggml/upstream.cu'))
            with self.assertRaises(ValueError):MODULE.rewrite_owned_compile(bad,old,new,driver,fresh)


if __name__=='__main__':unittest.main()
